import hashlib
import io
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict, Union

import numpy as np
import pillow_avif  # noqa: F401
from PIL import Image

from .tiles import (
    calculate_tile_based_psnr,
    detect_optimal_tile_size,
    precompute_tile_positions,
)


class IterationInfo(TypedDict):
    """Information about a single compression iteration."""

    quality: int
    perceptual_quality: float
    psnr: float
    error: float
    file_size: int


@dataclass
class CompressionResult:
    compressed: Image.Image
    original: Image.Image
    iterations_info: list[IterationInfo]
    selected_quality: int
    hash: str
    hash_type: str
    file_size: int
    fmt: str
    extra_save_args: dict
    tile_size: int
    filename: Optional[str]
    content_type: str

    def save_thumbnail(self, max_size: int, output: Optional[Union[str, io.BytesIO]] = None) -> 'CompressionResult':
        """Save a thumbnail version of the compressed image with a new maximum size."""
        return Pixiq.save_thumbnail(self, max_size, output)

    @property
    def iterations_count(self) -> int:
        return len(self.iterations_info)

    @property
    def file_size_kb(self) -> float:
        """Get the size of the compressed image in kilobytes."""
        return self.file_size / 1024

    @property
    def dimensions(self) -> tuple[int, int]:
        """Get the dimensions (width, height) of the compressed image."""
        return self.compressed.size

    @property
    def best_iteration(self) -> Optional[IterationInfo]:
        """Get information about the best iteration found."""
        if not self.iterations_info:
            return None
        return min(self.iterations_info, key=lambda x: x['error'])

    def save(self, output: Union[str, io.BytesIO]) -> None:
        """Save the compressed image to the specified output."""
        compressed_buffer, _ = Pixiq._compress_to_bytes(
            self.compressed,
            self.fmt,
            self.selected_quality,
            self.extra_save_args,
        )
        Pixiq._save_output(compressed_buffer, output)


class Pixiq:
    # Constants for quality conversion
    PSNR_TO_PERCEPTUAL_QUALITY_RATIO = 0.0282  # Empirical ratio to convert PSNR to perceptual quality

    # Default quality bounds
    DEFAULT_MIN_QUALITY = 1
    DEFAULT_MAX_QUALITY = 100
    DEFAULT_MAX_ITERATIONS = 5

    # Supported formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}
    FORMAT_MAP = {
        '.jpg': 'JPEG',
        '.jpeg': 'JPEG',
        '.png': 'PNG',
        '.webp': 'WEBP',
        '.avif': 'AVIF',
    }

    @staticmethod
    def compress(
        input: Image.Image,
        output: Optional[Union[str, io.BytesIO]] = None,
        perceptual_quality: float = 0.9,
        tolerance: float = 0.005,
        max_quality: Optional[int] = None,
        min_quality: Optional[int] = None,
        max_size: Optional[int] = None,
        max_iter: int = 5,
        format: Optional[str] = None,
        hash_type: str = 'sha256',
        tile_size: Optional[int] = None,
        top_tiles_count: int = 5,
    ) -> CompressionResult:
        # Input validation
        if not isinstance(input, Image.Image):
            raise TypeError('Input must be a PIL Image')
        if input.size[0] == 0 or input.size[1] == 0:
            raise ValueError('Image dimensions must be positive')
        if perceptual_quality < 0.0 or perceptual_quality > 1.0:
            raise ValueError('Perceptual quality must be between 0.0 and 1.0')
        if tolerance <= 0.0:
            raise ValueError('Tolerance must be positive')
        if max_quality is not None and (max_quality < 1 or max_quality > 100):
            raise ValueError('Max quality must be between 1 and 100')
        if min_quality is not None and (min_quality < 1 or min_quality > 100):
            raise ValueError('Min quality must be between 1 and 100')
        if max_quality is not None and min_quality is not None and max_quality < min_quality:
            raise ValueError('Max quality must be greater than or equal to min quality')
        if max_size is not None and max_size <= 0:
            raise ValueError('Max size must be positive')
        if max_iter <= 0:
            raise ValueError('Max iterations must be positive')
        if tile_size is not None and tile_size <= 0:
            raise ValueError('Tile size must be positive')
        if top_tiles_count <= 0:
            raise ValueError('Top tiles count must be positive')

        # Detect format first to determine alpha support
        fmt, extra_save_args = Pixiq._detect_format(input, format, output)

        # Preserve alpha channel if present and format supports it
        has_alpha = input.mode == 'RGBA' or (input.mode == 'P' and 'transparency' in input.info)
        supports_alpha = fmt.upper() in ('PNG', 'WEBP', 'AVIF')

        if has_alpha and supports_alpha:
            img = input.convert('RGBA')
        else:
            img = input.convert('RGB')

        # Save original image before applying max_size (for thumbnail creation)
        original_img = img.copy()

        if max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        orig_array = np.array(img)

        # Auto-detect tile size if not provided
        if tile_size is None:
            tile_size = detect_optimal_tile_size(orig_array)

        # Precompute tile positions and slices with most colors once
        selected_tile_indices, selected_tile_slices = precompute_tile_positions(orig_array, tile_size, top_tiles_count)

        low = min_quality if min_quality is not None else Pixiq.DEFAULT_MIN_QUALITY
        high = max_quality if max_quality is not None else Pixiq.DEFAULT_MAX_QUALITY
        best_buffer = None
        best_size = 0
        best_error = float('inf')
        best_quality = high
        iterations_info = []
        last_valid_quality = high  # For fallback

        iteration = 0
        while low <= high and iteration < max_iter:
            iteration += 1
            mid = (low + high) // 2

            buffer = Pixiq._compress_image_to_buffer(img, fmt, mid, extra_save_args)
            # Optimization: save buffer size via tell() (faster than getvalue())
            file_size = buffer.tell()
            buffer.seek(0)

            try:
                with Image.open(buffer) as comp_img:
                    comp = comp_img.convert(img.mode)
                    comp_array = np.array(comp)
            except (OSError, ValueError):
                # Skip invalid quality levels that can't be decoded
                high = mid - 1
                continue

            current_psnr = calculate_tile_based_psnr(
                orig_array, comp_array, tile_size, selected_tile_indices, selected_tile_slices
            )
            current_perceptual_quality = Pixiq.PSNR_TO_PERCEPTUAL_QUALITY_RATIO * current_psnr
            error = abs(current_perceptual_quality - perceptual_quality)

            iterations_info.append(
                {
                    'quality': mid,
                    'perceptual_quality': current_perceptual_quality,
                    'psnr': current_psnr,
                    'error': error,
                    'file_size': file_size,
                }
            )

            last_valid_quality = mid  # Update last valid quality

            if error < best_error:
                # Optimization: save buffer data once instead of creating a copy
                buffer_data = buffer.getvalue()
                best_buffer = io.BytesIO(buffer_data)
                best_size = file_size
                best_error = error
                best_quality = mid

            if error < tolerance:
                break

            if current_perceptual_quality < perceptual_quality:
                low = mid + 1
            else:
                high = mid - 1

        # Use the best buffer if available, otherwise compress again with last valid quality
        if best_buffer is not None:
            compressed_buffer = best_buffer
            file_size = best_size
        else:
            # Fallback: compress with last valid quality
            compressed_buffer, file_size = Pixiq._compress_to_bytes(img, fmt, last_valid_quality, extra_save_args)

        # Save buffer data BEFORE opening image (PIL may close the buffer)
        compressed_buffer.seek(0)
        buffer_data = compressed_buffer.getvalue()

        # Get compressed image and hash
        compressed_buffer_for_image = io.BytesIO(buffer_data)
        compressed_image = Image.open(compressed_buffer_for_image)
        compressed_copy = compressed_image.copy()
        final_hash = Pixiq.get_image_hash(compressed_image, hash_type=hash_type)
        compressed_image.close()

        # Save to the actual output if specified
        # Use saved buffer data instead of the buffer itself
        compressed_buffer_for_save = io.BytesIO(buffer_data)
        Pixiq._save_output(compressed_buffer_for_save, output)

        # Generate filename and content_type
        filename = Pixiq._generate_filename(input, output, fmt)
        content_type = Pixiq._get_content_type(fmt)

        result = CompressionResult(
            compressed=compressed_copy,
            original=original_img,
            iterations_info=iterations_info,
            selected_quality=best_quality if best_buffer is not None else last_valid_quality,
            hash=final_hash,
            file_size=file_size,
            fmt=fmt.lower(),
            extra_save_args=extra_save_args,
            hash_type=hash_type,
            tile_size=tile_size,
            filename=filename,
            content_type=content_type,
        )
        return result

    @staticmethod
    def _detect_format_from_source(image: Image.Image) -> str:
        """Detect format from image filename or format attribute."""
        filename = getattr(image, 'filename', None)
        if filename:
            ext = os.path.splitext(filename)[1].lower()
            return Pixiq.FORMAT_MAP.get(ext, 'JPEG')
        elif hasattr(image, 'format') and image.format:
            return image.format.upper()
        else:
            return 'JPEG'

    @staticmethod
    def _generate_filename(image: Image.Image, output: Optional[Union[str, io.BytesIO]], fmt: str) -> Optional[str]:
        """Generate output filename based on input image filename and format."""
        if isinstance(output, str):
            path = Path(output)
            return f'{path.stem}{path.suffix}'  # Сохраняет оригинальное имя+расширение

        original_filename = getattr(image, 'filename', None)
        if not original_filename:
            return None

        name = Path(original_filename).stem
        return f'{name}.{fmt.lower()}'

    @staticmethod
    def _get_content_type(fmt: str) -> str:
        """Get MIME content type for the given format using mimetypes."""
        content_type, _ = mimetypes.guess_type(f'dummy.{fmt.lower()}')
        return content_type or 'image/jpeg'

    @staticmethod
    def _detect_format(
        image: Image.Image,
        format: Optional[str] = None,
        output: Optional[Union[str, io.BytesIO]] = None,
    ) -> tuple[str, dict]:
        """Detect format from image or parameter and return format string and save arguments."""
        if format is not None:
            fmt = format.upper()
        elif isinstance(output, str):
            # Try to get format from output path first, then fallback to image
            ext = os.path.splitext(output)[1].lower()
            fmt = Pixiq.FORMAT_MAP.get(ext)
            if fmt is None:
                fmt = Pixiq._detect_format_from_source(image)
        else:
            # No output path specified, detect from image
            fmt = Pixiq._detect_format_from_source(image)

        # Validate format
        if fmt is None or fmt not in {'JPEG', 'PNG', 'WEBP', 'AVIF'}:
            fmt = 'JPEG'  # Default fallback

        # Normalize format
        if fmt == 'JPG':
            fmt = 'JPEG'

        # Format-specific save parameters
        extra_save_args = {}
        if fmt == 'JPEG':
            extra_save_args = dict(optimize=True, progressive=True)
        elif fmt == 'WEBP':
            extra_save_args = dict(method=4, lossless=True)
        elif fmt == 'AVIF':
            extra_save_args = dict(speed=6)
        elif fmt == 'PNG':
            extra_save_args = dict(optimize=True, compress_level=5)

        return fmt, extra_save_args

    @staticmethod
    def _compress_image_to_buffer(
        image: Image.Image,
        fmt: str,
        quality: int,
        extra_save_args: dict,
    ) -> io.BytesIO:
        """Compress image to BytesIO buffer with format-specific handling."""
        buffer = io.BytesIO()
        try:
            if fmt.upper() == 'PNG':
                colors = Pixiq._quality_to_colors(quality)
                quantized = image.quantize(colors=colors)
                quantized.save(buffer, fmt, **extra_save_args)
            else:
                image.save(buffer, fmt, quality=quality, **extra_save_args)
        except (OSError, ValueError) as e:
            raise OSError(f'Failed to compress image: {e}') from e
        return buffer

    @staticmethod
    def _quality_to_colors(quality: int) -> int:
        """Convert quality (1-100) to number of colors for PNG quantize.

        Maps quality to color count using exponential formula:
        quality 1 -> 2 colors, quality 100 -> 256 colors

        Args:
            quality: Quality value from 1 to 100

        Returns:
            Number of colors (2-256)
        """
        x = max(0, min(100, quality))
        p = 1 + 7 * (x / 100)  # Exponential mapping: 2^1 to 2^8
        return int(2**p)

    @staticmethod
    def _compress_to_bytes(
        final_image: Image.Image,
        fmt: str,
        quality: int,
        extra_save_args: dict,
    ) -> tuple[io.BytesIO, int]:
        """Compress image to BytesIO buffer and return it with file size."""
        buffer = Pixiq._compress_image_to_buffer(final_image, fmt, quality, extra_save_args)
        # Use tell() to get size (faster than getvalue())
        file_size = buffer.tell()
        buffer.seek(0)
        return buffer, file_size

    @staticmethod
    def _save_output(
        compressed_buffer: io.BytesIO,
        output: Optional[Union[str, io.BytesIO]],
    ) -> None:
        """Save compressed data from BytesIO to the specified output."""
        if output is not None:
            try:
                if isinstance(output, str):
                    # Save buffer to file
                    with open(output, 'wb') as f:
                        f.write(compressed_buffer.getvalue())
                elif isinstance(output, io.BytesIO):
                    # Optimization: save data once instead of double getvalue() call
                    buffer_data = compressed_buffer.getvalue()
                    output.seek(0)
                    output.truncate(0)
                    output.write(buffer_data)
                    output.seek(0)
                else:
                    raise TypeError('Output must be a file path (str) or BytesIO buffer')
            except Exception as e:
                raise OSError(f'Failed to save image: {e}') from e

    @staticmethod
    def save_thumbnail(
        result: CompressionResult,
        max_size: int,
        output: Optional[Union[str, io.BytesIO]] = None,
    ) -> CompressionResult:
        """Save a thumbnail version using the original image with preserved compression parameters."""
        # Input validation
        if not isinstance(result, CompressionResult):
            raise TypeError('Result must be a CompressionResult instance')

        if max_size <= 0:
            raise ValueError('Max size must be positive')

        # Use original image instead of compressed
        original_img = result.original

        # Create a copy of the original image and resize it
        resized_image = original_img.copy()
        if max_size < max(resized_image.size):
            resized_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Compress resized image to bytes using the same quality and format as original compression
        compressed_buffer, file_size = Pixiq._compress_to_bytes(
            resized_image,
            result.fmt,
            result.selected_quality,
            result.extra_save_args,
        )

        # Get compressed image and hash
        with Image.open(compressed_buffer) as compressed_image:
            compressed_copy = compressed_image.copy()
            final_hash = Pixiq.get_image_hash(compressed_image, hash_type=result.hash_type)

        # Create new CompressionResult with the resized image
        # Save original image (before resizing) for possible future thumbnails
        new_result = CompressionResult(
            compressed=compressed_copy,
            original=original_img,  # Save original image
            iterations_info=[],  # No iteration info for resizing
            selected_quality=result.selected_quality,
            hash=final_hash,
            fmt=result.fmt,
            file_size=file_size,
            extra_save_args=result.extra_save_args.copy(),
            hash_type=result.hash_type,
            tile_size=result.tile_size,
            filename=result.filename,
            content_type=result.content_type,
        )

        # Save to output if specified
        Pixiq._save_output(compressed_buffer, output)

        return new_result

    @staticmethod
    def get_image_hash(image: Image.Image, hash_type: str = 'sha256') -> str:
        """Calculate hash of image pixel data.

        Args:
            image: PIL Image to hash
            hash_type: Hash algorithm name (e.g., 'sha256', 'sha1')

        Returns:
            Hexadecimal hash string

        Raises:
            ValueError: If hash_type is not supported
        """
        if hash_type not in hashlib.algorithms_available:
            raise ValueError(f"Hash type '{hash_type}' not supported.")
        hasher = hashlib.new(hash_type)
        hasher.update(image.tobytes())
        return hasher.hexdigest()

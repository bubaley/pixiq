import hashlib
import io
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import numpy as np
import pillow_avif  # noqa: F401
from PIL import Image

from .psnr import psnr


@dataclass
class CompressionResult:
    compressed: Image.Image
    iterations_count: int
    iterations_info: list[dict]
    selected_quality: int
    hash: str
    fmt: str
    extra_save_args: dict

    def save_thumbnail(self, max_size: int, output: Optional[Union[str, io.BytesIO]] = None) -> 'CompressionResult':
        """Save a thumbnail version of the compressed image with a new maximum size."""
        return Pixiq.save_thumbnail(self, max_size, output)

    @property
    def file_size_bytes(self) -> int:
        """Get the size of the compressed image in bytes."""
        return len(self.compressed.tobytes())

    @property
    def file_size_kb(self) -> float:
        """Get the size of the compressed image in kilobytes."""
        return self.file_size_bytes / 1024

    @property
    def dimensions(self) -> tuple[int, int]:
        """Get the dimensions (width, height) of the compressed image."""
        return self.compressed.size

    @property
    def last_iteration(self) -> Optional[dict]:
        """Get information about the last iteration performed."""
        return self.iterations_info[-1] if self.iterations_info else None

    @property
    def best_iteration(self) -> Optional[dict]:
        """Get information about the best iteration found."""
        if not self.iterations_info:
            return None
        # Find iteration with minimum error
        return min(self.iterations_info, key=lambda x: x.get('error', float('inf')))

    def save(self, output: Union[str, io.BytesIO]) -> None:
        """Save the compressed image to the specified output."""
        Pixiq._save_output(
            self.compressed,
            self.fmt,
            self.selected_quality,
            self.extra_save_args,
            output,
        )


class Pixiq:
    # Constants for quality conversion
    PSNR_TO_PERCEPTUAL_QUALITY_RATIO = 0.025  # Empirical ratio to convert PSNR to perceptual quality

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
        perceptual_quality: float = 0.95,
        tolerance: float = 0.005,
        max_quality: Optional[int] = None,
        min_quality: Optional[int] = None,
        max_size: Optional[int] = None,
        max_iter: int = 5,
        format: Optional[str] = None,
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

        img = input.convert('RGB')
        if max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        orig_array = np.array(img)

        # Detect format and get save parameters
        fmt, extra_save_args = Pixiq._detect_format(input, format, output)

        low = min_quality if min_quality is not None else Pixiq.DEFAULT_MIN_QUALITY
        high = max_quality if max_quality is not None else Pixiq.DEFAULT_MAX_QUALITY
        best_image = None
        best_error = float('inf')
        best_quality = high
        iterations_info = []

        iteration = 0
        while low <= high and iteration < max_iter:
            iteration += 1
            mid = (low + high) // 2

            buffer = io.BytesIO()
            img.save(buffer, fmt, quality=mid, **extra_save_args)
            buffer.seek(0)

            try:
                comp = Image.open(buffer).convert('RGB')
            except Exception as e:
                # Skip invalid quality levels that can't be decoded
                print(f'Warning: Failed to decode image with quality {mid}: {e}')
                high = mid - 1
                continue

            comp_array = np.array(comp)
            current_psnr = psnr(orig_array, comp_array)
            current_perceptual_quality = Pixiq.PSNR_TO_PERCEPTUAL_QUALITY_RATIO * current_psnr
            error = abs(current_perceptual_quality - perceptual_quality)
            size_kb = len(buffer.getvalue()) / 1024

            iteration_data = {
                'quality': mid,
                'perceptual_quality': current_perceptual_quality,
                'psnr': current_psnr,
                'error': error,
                'size_kb': size_kb,
            }
            iterations_info.append(iteration_data)

            if error < best_error:
                best_image = comp
                best_error = error
                best_quality = mid

            if error < tolerance:
                break

            if current_perceptual_quality < perceptual_quality:
                low = mid + 1
            else:
                high = mid - 1

        final_image = best_image if best_image else img

        # Calculate hash from the compressed buffer to avoid re-encoding
        compressed_buffer = io.BytesIO()
        final_image.save(compressed_buffer, fmt, quality=best_quality, **extra_save_args)
        compressed_buffer.seek(0)
        image_hash = Pixiq.get_file_hash(compressed_buffer)

        result = CompressionResult(
            compressed=final_image,
            iterations_count=iteration,
            iterations_info=iterations_info,
            selected_quality=best_quality,
            hash=image_hash,
            fmt=fmt,
            extra_save_args=extra_save_args,
        )
        # Save to output if specified
        Pixiq._save_output(final_image, fmt, best_quality, extra_save_args, output)
        return result

    @staticmethod
    def _detect_format_from_source(image: Image.Image) -> str:
        """Detect format from image filename or format attribute."""
        import os

        filename = getattr(image, 'filename', None)
        if filename:
            ext = os.path.splitext(filename)[1].lower()
            return Pixiq.FORMAT_MAP.get(ext, 'JPEG')
        elif hasattr(image, 'format') and image.format:
            return image.format.upper()
        else:
            return 'JPEG'

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
            import os

            ext = os.path.splitext(output)[1].lower()
            fmt = Pixiq.FORMAT_MAP.get(ext)
            if fmt is None:
                fmt = Pixiq._detect_format_from_source(image)
        else:
            # No output path specified, detect from image
            fmt = Pixiq._detect_format_from_source(image)

        # Normalize format
        if fmt == 'JPG':
            fmt = 'JPEG'

        # Format-specific save parameters
        extra_save_args = {}
        if fmt == 'JPEG':
            extra_save_args = dict(optimize=True, progressive=True)
        elif fmt == 'WEBP':
            extra_save_args = dict(method=6)
        elif fmt == 'AVIF':
            extra_save_args = dict(speed=6)

        return fmt, extra_save_args

    @staticmethod
    def _save_output(
        final_image: Image.Image,
        fmt: str,
        quality: int,
        extra_save_args: dict,
        output: Optional[Union[str, io.BytesIO]],
    ) -> None:
        """Save the compressed image to the specified output."""
        if output is not None:
            try:
                if isinstance(output, str):
                    # Save to file
                    final_image.save(output, fmt, quality=quality, **extra_save_args)
                elif isinstance(output, io.BytesIO):
                    # Save to buffer
                    output.seek(0)
                    output.truncate(0)
                    final_image.save(output, fmt, quality=quality, **extra_save_args)
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
        """Save a thumbnail version of a compressed image with a new maximum size."""
        # Input validation
        if not isinstance(result, CompressionResult):
            raise TypeError('Result must be a CompressionResult instance')

        if max_size <= 0:
            raise ValueError('Max size must be positive')

        if max_size > max(result.compressed.size):
            # No need to resize if new size is larger than current
            return result

        # Create a copy of the compressed image and resize it
        resized_image = result.compressed.copy()
        resized_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Calculate hash from compressed buffer to avoid re-encoding
        compressed_buffer = io.BytesIO()
        resized_image.save(
            compressed_buffer,
            result.fmt,
            quality=result.selected_quality,
            **result.extra_save_args,
        )
        compressed_buffer.seek(0)
        new_hash = Pixiq.get_file_hash(compressed_buffer)

        # Create new CompressionResult with the resized image
        # Note: iterations_count and iterations_info are not copied since resizing doesn't involve quality search
        new_result = CompressionResult(
            compressed=resized_image,
            iterations_count=0,  # No iterations performed for resizing
            iterations_info=[],  # No iteration info for resizing
            selected_quality=result.selected_quality,  # Keep original quality
            hash=new_hash,
            fmt=result.fmt,
            extra_save_args=result.extra_save_args.copy(),
        )

        # Save to output if specified
        Pixiq._save_output(
            resized_image,
            result.fmt,
            result.selected_quality,
            result.extra_save_args,
            output,
        )

        return new_result

    @staticmethod
    def get_file_hash(file, block_size: int = 65536) -> str:
        hasher = hashlib.sha256()
        for buf in iter(partial(file.read, block_size), b''):
            hasher.update(buf)
        return hasher.hexdigest()

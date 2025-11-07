import hashlib
import io
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pillow_avif  # noqa: F401
from PIL import Image

from .photo_score import photo_score
from .psnr import psnr


@dataclass
class CompressionResult:
    compressed: Image.Image
    iterations_info: list[dict]
    selected_quality: int
    hash: str
    hash_type: str
    file_size: int
    fmt: str
    extra_save_args: dict
    tile_size_ratio: float

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
    def best_iteration(self) -> Optional[dict]:
        """Get information about the best iteration found."""
        if not self.iterations_info:
            return None
        return min(self.iterations_info, key=lambda x: x.get('error', float('inf')))

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
    PSNR_TO_PERCEPTUAL_QUALITY_RATIO = 0.02767  # Empirical ratio to convert PSNR to perceptual quality

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
        hash_type: str = 'sha256',
        tile_size_ratio: Optional[float] = None,
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
        if tile_size_ratio is not None and (tile_size_ratio <= 0.0 or tile_size_ratio > 1.0):
            raise ValueError('Tile size ratio must be between 0.0 and 1.0')
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

        if max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        orig_array = np.array(img)

        # Auto-detect tile size ratio if not provided
        if tile_size_ratio is None:
            tile_size_ratio = Pixiq._detect_optimal_tile_size(orig_array)

        # Precompute tile positions with most colors once
        selected_tile_indices = Pixiq._precompute_tile_positions(orig_array, tile_size_ratio, top_tiles_count)

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

            buffer = io.BytesIO()
            img.save(buffer, fmt, quality=mid, **extra_save_args)
            file_size = buffer.tell()
            buffer.seek(0)

            try:
                comp = Image.open(buffer)
                comp = comp.convert(img.mode)
            except Exception as e:
                # Skip invalid quality levels that can't be decoded
                print(f'Warning: Failed to decode image with quality {mid}: {e}')
                high = mid - 1
                continue

            comp_array = np.array(comp)
            current_psnr = Pixiq._calculate_tile_based_psnr(
                orig_array, comp_array, tile_size_ratio, selected_tile_indices
            )
            # current_psnr = psnr(orig_array, comp_array)
            current_perceptual_quality = Pixiq.PSNR_TO_PERCEPTUAL_QUALITY_RATIO * current_psnr
            print(f'current_perceptual_quality: {current_perceptual_quality} current_psnr: {current_psnr}')
            error = abs(current_perceptual_quality - perceptual_quality)

            iterations_info.append(
                {
                    'quality': mid,
                    'perceptual_quality': current_perceptual_quality,
                    'psnr': current_psnr,
                    'error': error,
                    'file_size': file_size,
                    'hash': Pixiq.get_image_hash(comp, hash_type=hash_type),
                }
            )

            last_valid_quality = mid  # Update last valid quality

            if error < best_error:
                best_buffer = io.BytesIO(buffer.getvalue())  # Copy buffer data
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

        # Get compressed image and hash
        with Image.open(compressed_buffer) as compressed_image:
            compressed_copy = compressed_image.copy()
            final_hash = Pixiq.get_image_hash(compressed_image, hash_type=hash_type)

        # Save to the actual output if specified
        Pixiq._save_output(compressed_buffer, output)

        result = CompressionResult(
            compressed=compressed_copy,
            iterations_info=iterations_info,
            selected_quality=best_quality if best_buffer is not None else last_valid_quality,
            hash=final_hash,
            file_size=file_size,
            fmt=fmt.lower(),
            extra_save_args=extra_save_args,
            hash_type=hash_type,
            tile_size_ratio=tile_size_ratio,
        )
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
        # PNG doesn't use quality parameter

        return fmt, extra_save_args

    @staticmethod
    def _compress_to_bytes(
        final_image: Image.Image,
        fmt: str,
        quality: int,
        extra_save_args: dict,
    ) -> tuple[io.BytesIO, int]:
        """Compress image to BytesIO buffer and return it with file size."""
        buffer = io.BytesIO()
        try:
            final_image.save(buffer, fmt, quality=quality, **extra_save_args)
            file_size = buffer.tell()
            buffer.seek(0)
            return buffer, file_size
        except Exception as e:
            raise OSError(f'Failed to compress image: {e}') from e

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
                    # Copy buffer to output BytesIO
                    output.seek(0)
                    output.truncate(0)
                    output.write(compressed_buffer.getvalue())
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
        if max_size < max(resized_image.size):
            resized_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Compress resized image to bytes
        compressed_buffer, file_size = Pixiq._compress_to_bytes(
            resized_image,
            result.fmt,
            result.selected_quality,
            result.extra_save_args,
        )
        image = Image.open(compressed_buffer)

        # Create new CompressionResult with the resized image
        # Note: iterations_count and iterations_info are not copied since resizing doesn't involve quality search
        new_result = CompressionResult(
            compressed=image,
            iterations_info=[],  # No iteration info for resizing
            selected_quality=result.selected_quality,  # Keep original quality
            hash=Pixiq.get_image_hash(image, hash_type=result.hash_type),
            fmt=result.fmt,
            file_size=file_size,
            extra_save_args=result.extra_save_args.copy(),
            hash_type=result.hash_type,
            tile_size_ratio=result.tile_size_ratio,
        )

        # Save to output if specified
        Pixiq._save_output(compressed_buffer, output)

        return new_result

    @staticmethod
    def _split_into_tiles(
        image_array: np.ndarray, tile_size_ratio: float = 0.1, overlap: bool = True
    ) -> list[tuple[np.ndarray, tuple[int, int]]]:
        """Split image into tiles of size tile_size_ratio.

        Args:
            image_array: Image array (H, W, C)
            tile_size_ratio: Tile size ratio (default 0.1 = 10%)
            overlap: If True, tiles overlap for more coverage

        Returns:
            List of (tile, (row, col)) tuples
        """
        height, width = image_array.shape[:2]
        tile_height = max(1, int(height * tile_size_ratio))
        tile_width = max(1, int(width * tile_size_ratio))
        step_height = tile_height // 2 if overlap else tile_height
        step_width = tile_width // 2 if overlap else tile_width

        tiles = []
        for row in range(0, height - tile_height + 1, step_height):
            for col in range(0, width - tile_width + 1, step_width):
                tile = image_array[row : row + tile_height, col : col + tile_width]
                if tile.size > 0:
                    tiles.append((tile, (row, col)))
        return tiles

    @staticmethod
    def _count_unique_colors(tile: np.ndarray) -> int:
        """Count unique colors in a tile.

        Args:
            tile: Tile array (H, W, C)

        Returns:
            Number of unique colors
        """
        if len(tile.shape) == 3:
            pixels = tile.reshape(-1, tile.shape[2])
            dtype = np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1]))
            return len(np.unique(pixels.view(dtype)))
        return len(np.unique(tile.flatten()))

    @staticmethod
    def _select_tiles_with_most_colors(
        tiles: list[tuple[np.ndarray, tuple[int, int]]], top_n: Optional[int] = 5
    ) -> list[tuple[np.ndarray, int]]:
        """Select tiles with most unique colors, avoiding adjacent tiles.

        Args:
            tiles: List of (tile, coords) tuples
            top_n: Number of tiles to select (None = all)

        Returns:
            Sorted list of (tile, index) tuples
        """
        tiles_with_color_count = [
            (idx, tile, coords, Pixiq._count_unique_colors(tile)) for idx, (tile, coords) in enumerate(tiles)
        ]
        tiles_with_color_count.sort(key=lambda x: x[3], reverse=True)

        # Select tiles avoiding adjacent ones with similar color counts
        selected = []
        if not tiles:
            return []

        # Calculate minimum distance based on tile size
        tile_height, tile_width = tiles[0][0].shape[:2]
        min_distance = max(tile_height, tile_width) * 1.5  # Minimum distance between tile centers

        for idx, tile, coords, color_count in tiles_with_color_count:
            if top_n is not None and len(selected) >= top_n:
                break

            # Check if tile is far enough from already selected tiles
            is_far_enough = True
            for sel_idx, sel_tile, sel_coords, sel_color_count in selected:
                row1, col1 = coords
                row2, col2 = sel_coords
                distance = ((row1 - row2) ** 2 + (col1 - col2) ** 2) ** 0.5

                # If tiles are close and have similar color counts, skip
                if (
                    distance < min_distance
                    and abs(color_count - sel_color_count) < max(color_count, sel_color_count) * 0.15
                ):
                    is_far_enough = False
                    break

            if is_far_enough:
                selected.append((idx, tile, coords, color_count))

        return [(tile, idx) for idx, tile, _, _ in selected]

    @staticmethod
    def _detect_optimal_tile_size(image_array: np.ndarray) -> float:
        """Detect optimal tile size: smaller for UI, larger for photos.

        Uses photo_score to determine if image is more UI-like (low score)
        or photo-like (high score) and adjusts tile size accordingly.
        """
        # Вычисляем photo_score для определения типа изображения
        # photo_score возвращает 0.0 для UI-like и 1.0 для photo-like
        start_time = datetime.now()
        score = photo_score(image_array, debug=False)
        end_time = datetime.now()
        print(f'Photo score time: {end_time - start_time}')

        # Инвертируем score для получения UI_score (высокий = UI, низкий = фото)
        ui_score = 1.0 - score

        # === Mapping to tile size based on photo_score ===
        # UI (высокий ui_score): мелкие тайлы для точности
        # Photo (низкий ui_score): крупные тайлы для эффективности
        print(f'ui_score: {ui_score}')
        if ui_score > 0.7:
            return 0.05  # Чистый UI: мелкие тайлы
        elif ui_score > 0.5:
            return 0.08  # Смешанный / простой UI
        elif ui_score > 0.3:
            return 0.2  # Фото с чёткими краями
        else:
            return 0.3  # Сложные фото, градиенты

    @staticmethod
    def _precompute_tile_positions(
        image_array: np.ndarray, tile_size_ratio: float, top_tiles_count: int
    ) -> list[tuple[int, int]]:
        """Precompute positions of tiles with most colors.

        Args:
            image_array: Image array (H, W, C)
            tile_size_ratio: Tile size ratio
            top_tiles_count: Number of tiles to select

        Returns:
            List of (row, col) coordinates
        """
        tiles = Pixiq._split_into_tiles(image_array, tile_size_ratio)
        if not tiles:
            return []
        selected_tiles = Pixiq._select_tiles_with_most_colors(tiles, top_n=top_tiles_count)
        if not selected_tiles:
            return []
        return [tiles[idx][1] for _, idx in selected_tiles]

    @staticmethod
    def _calculate_tile_based_psnr(
        original: np.ndarray,
        compressed: np.ndarray,
        tile_size_ratio: float,
        selected_tile_coords: list[tuple[int, int]],
    ) -> float:
        """Calculate PSNR using precomputed tile positions.

        Args:
            original: Original image array
            compressed: Compressed image array
            tile_size_ratio: Tile size ratio
            selected_tile_coords: List of (row, col) coordinates

        Returns:
            Minimum PSNR among selected tiles
        """
        if not selected_tile_coords:
            return psnr(original, compressed)

        height, width = original.shape[:2]
        tile_height = max(1, int(height * tile_size_ratio))
        tile_width = max(1, int(width * tile_size_ratio))

        psnr_values = []
        for row, col in selected_tile_coords:
            orig_tile = original[row : row + tile_height, col : col + tile_width]
            comp_tile = compressed[row : row + tile_height, col : col + tile_width]
            if orig_tile.shape == comp_tile.shape and orig_tile.size > 0:
                psnr_values.append(psnr(orig_tile, comp_tile))

        return min(psnr_values) if psnr_values else psnr(original, compressed)

    @staticmethod
    def get_image_hash(image: Image.Image, hash_type: str = 'sha256') -> str:
        if hash_type not in hashlib.algorithms_available:
            raise ValueError(f"Hash type '{hash_type}' not supported.")
        hasher = hashlib.new(hash_type)
        hasher.update(image.tobytes())
        return hasher.hexdigest()

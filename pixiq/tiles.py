from typing import Optional

import numpy as np

from .photo_score import photo_score
from .psnr import psnr


def split_into_tiles(
    image_array: np.ndarray, tile_size: int, overlap: bool = True
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Split image into tiles of specified size.

    Args:
        image_array: Image array (H, W, C)
        tile_size: Tile size in pixels
        overlap: If True, tiles overlap for more coverage

    Returns:
        List of (tile, (row, col)) tuples
    """
    height, width = image_array.shape[:2]
    tile_height = max(1, min(tile_size, height))
    tile_width = max(1, min(tile_size, width))
    step_height = tile_height // 2 if overlap else tile_height
    step_width = tile_width // 2 if overlap else tile_width

    tiles = []
    for row in range(0, height - tile_height + 1, step_height):
        for col in range(0, width - tile_width + 1, step_width):
            tile = image_array[row : row + tile_height, col : col + tile_width]
            if tile.size > 0:
                tiles.append((tile, (row, col)))
    return tiles


def count_unique_colors(tile: np.ndarray) -> int:
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


def complexity_score(tile: np.ndarray) -> float:
    """
    Calculate entropy of pixel distribution in tile (higher = more complex).
    Works with numpy only, without scipy.
    Optimized to use bincount where possible.
    """
    if tile.size == 0:
        return 0.0

    if len(tile.shape) == 3:
        # For RGB: convert each pixel to unique number (0..2^24-1)
        # This is faster than histogramdd
        r, g, b = tile[..., 0], tile[..., 1], tile[..., 2]
        pixels = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
        flat = pixels.ravel()
    else:
        # Grayscale or other format
        flat = tile.ravel().astype(np.uint32)

    # Optimization: for RGB always use bincount if possible
    # bincount works only for non-negative integers and requires continuous range
    max_val = flat.max()
    if max_val < 1_000_000 and max_val < len(flat) * 10:  # bincount is efficient if range is not too large
        counts = np.bincount(flat)
        # Filter zeros
        counts = counts[counts > 0]
    else:
        # For large ranges use unique + count
        unique_vals, counts = np.unique(flat, return_counts=True)
        counts = counts[counts > 0]

    if len(counts) <= 1:
        return 0.0

    # Probabilities
    total = counts.sum()
    probs = counts / total

    # Entropy: -sum(p * log2(p))
    # Use np.log2, avoid log(0)
    entropy_val = -np.sum(probs * np.log2(probs))
    return float(entropy_val)


def select_tiles_with_most_colors(
    tiles: list[tuple[np.ndarray, tuple[int, int]]], top_n: Optional[int] = 5
) -> list[tuple[np.ndarray, int]]:
    """Select tiles with most unique colors, avoiding adjacent tiles.

    Args:
        tiles: List of (tile, coords) tuples
        top_n: Number of tiles to select (None = all)

    Returns:
        Sorted list of (tile, index) tuples
    """
    tiles_with_color_count = [(idx, tile, coords, complexity_score(tile)) for idx, (tile, coords) in enumerate(tiles)]
    tiles_with_color_count.sort(key=lambda x: x[3], reverse=True)

    # Select tiles avoiding adjacent ones with similar color counts
    selected = []
    if not tiles:
        return []

    # Calculate minimum distance based on tile size
    tile_height, tile_width = tiles[0][0].shape[:2]
    min_distance_sq = (max(tile_height, tile_width) * 1.5) ** 2  # Squared distance for faster comparison

    for idx, tile, coords, color_count in tiles_with_color_count:
        if top_n is not None and len(selected) >= top_n:
            break

        # Check if tile is far enough from already selected tiles
        # Optimization: use squared distance to avoid sqrt
        is_far_enough = True
        row1, col1 = coords
        for sel_idx, sel_tile, sel_coords, sel_color_count in selected:
            row2, col2 = sel_coords
            distance_sq = (row1 - row2) ** 2 + (col1 - col2) ** 2

            # If tiles are close and have similar color counts, skip
            if (
                distance_sq < min_distance_sq
                and abs(color_count - sel_color_count) < max(color_count, sel_color_count) * 0.15
            ):
                is_far_enough = False
                break

        if is_far_enough:
            selected.append((idx, tile, coords, color_count))

    return [(tile, idx) for idx, tile, _, _ in selected]


def detect_optimal_tile_size(image_array: np.ndarray) -> int:
    """Detect optimal tile size: smaller for UI, larger for photos.

    Uses photo_score to determine if image is more UI-like (low score)
    or photo-like (high score) and adjusts tile size accordingly.
    """
    score = photo_score(image_array, debug=False)
    ui_score = 1.0 - score  # Invert: high = UI, low = photo

    height, width = image_array.shape[:2]
    max_dimension = max(height, width)

    if ui_score > 0.7:
        ratio = 0.02
    elif ui_score > 0.6:
        ratio = 0.04
    elif ui_score > 0.5:
        ratio = 0.06
    elif ui_score > 0.4:
        ratio = 0.1
    elif ui_score > 0.3:
        ratio = 0.2
    else:
        ratio = 0.3
    return int(max_dimension * ratio)


def precompute_tile_positions(
    image_array: np.ndarray, tile_size: int, top_tiles_count: int
) -> tuple[list[tuple[int, int]], list[tuple[slice, slice]]]:
    """Precompute positions and slices of tiles with most colors.

    Args:
        image_array: Image array (H, W, C)
        tile_size: Tile size in pixels
        top_tiles_count: Number of tiles to select

    Returns:
        Tuple of (list of (row, col) coordinates, list of (row_slice, col_slice) tuples)
    """
    tiles = split_into_tiles(image_array, tile_size)
    if not tiles:
        return [], []
    selected_tiles = select_tiles_with_most_colors(tiles, top_n=top_tiles_count)
    if not selected_tiles:
        return [], []

    height, width = image_array.shape[:2]
    tile_height = max(1, min(tile_size, height))
    tile_width = max(1, min(tile_size, width))

    coords = []
    slices = []
    for _, idx in selected_tiles:
        row, col = tiles[idx][1]
        coords.append((row, col))
        # Precompute slice objects for fast access
        row_slice = slice(row, row + tile_height)
        col_slice = slice(col, col + tile_width)
        slices.append((row_slice, col_slice))

    return coords, slices


def calculate_tile_based_psnr(
    original: np.ndarray,
    compressed: np.ndarray,
    tile_size: int,
    selected_tile_coords: list[tuple[int, int]],
    selected_tile_slices: Optional[list[tuple[slice, slice]]] = None,
) -> float:
    """Calculate PSNR using precomputed tile positions and slices.

    Args:
        original: Original image array
        compressed: Compressed image array
        tile_size: Tile size in pixels
        selected_tile_coords: List of (row, col) coordinates
        selected_tile_slices: Optional list of (row_slice, col_slice) tuples for faster access

    Returns:
        Minimum PSNR among selected tiles
    """
    if not selected_tile_coords:
        return psnr(original, compressed)

    # Use precomputed slices if available, otherwise compute on the fly
    if selected_tile_slices is not None:
        psnr_values = []
        for row_slice, col_slice in selected_tile_slices:
            orig_tile = original[row_slice, col_slice]
            comp_tile = compressed[row_slice, col_slice]
            if orig_tile.shape == comp_tile.shape and orig_tile.size > 0:
                psnr_values.append(psnr(orig_tile, comp_tile))
    else:
        # Fallback: compute slices on the fly (old behavior for backward compatibility)
        height, width = original.shape[:2]
        tile_height = max(1, min(tile_size, height))
        tile_width = max(1, min(tile_size, width))
        psnr_values = []
        for row, col in selected_tile_coords:
            orig_tile = original[row : row + tile_height, col : col + tile_width]
            comp_tile = compressed[row : row + tile_height, col : col + tile_width]
            if orig_tile.shape == comp_tile.shape and orig_tile.size > 0:
                psnr_values.append(psnr(orig_tile, comp_tile))

    return min(psnr_values) if psnr_values else psnr(original, compressed)

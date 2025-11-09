import numpy as np
from PIL import Image, ImageFilter

# Constants for photo score calculation
RESIZE_SIZE = 256
QUANTIZATION_LEVELS = 31
ENTROPY_BINS = 64
ENTROPY_MIN_PROB = 1e-12
AUTO_CORR_PATCH_SIZE = 32
MAX_PIXEL_VALUE = 255.0

# Score weights
WEIGHT_CONTRAST = 2.8
WEIGHT_SATURATION = 2.2
WEIGHT_ENTROPY = 0.8
WEIGHT_TEXTURE = 0.006
WEIGHT_COLORS = 0.9

# Score thresholds and limits
TEXTURE_VAR_MAX = 150.0
MAX_UNIQUE_COLORS = 3500.0
EDGE_DENSITY_THRESHOLD = 0.35
SAT_STD_THRESHOLD = 0.03
CONTRAST_THRESHOLD = 0.08

# Penalties
PENALTY_HIGH_EDGE_DENSITY = 1.3
PENALTY_LOW_SAT_STD = 1.1
PENALTY_LOW_CONTRAST = 1.6

# Grid detection
GRID_RATIO_THRESHOLD = 5.0
GRID_PENALTY_MULTIPLIER = 0.04

# Normalization
SCORE_OFFSET = 1.0
SCORE_DIVISOR = 7.0


def photo_score(image_array: np.ndarray, debug: bool = False) -> float:
    """Calculate photo score from numpy array (0.0 = UI-like, 1.0 = photo-like).

    Args:
        image_array: Image array (H, W, C) with values in range [0, 255]
        debug: If True, print debug information

    Returns:
        Photo score normalized to [0.0, 1.0]
    """
    # Normalize and resize to RESIZE_SIZE x RESIZE_SIZE for consistency
    if len(image_array.shape) == 3:
        img = Image.fromarray(image_array.astype(np.uint8))
        img = img.convert('RGB').resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / MAX_PIXEL_VALUE
    else:
        img = Image.fromarray(image_array.astype(np.uint8), mode='L')
        img = img.resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / MAX_PIXEL_VALUE
        arr = np.stack([arr, arr, arr], axis=2)

    gray = arr.mean(axis=2)

    # Feature extraction
    contrast = gray.std()

    sat_map = np.std(arr, axis=2)
    sat_mean = sat_map.mean()
    sat_std = sat_map.std()

    quant = np.round(arr * QUANTIZATION_LEVELS).astype(np.uint8)
    unique_colors = len(np.unique(quant.reshape(-1, 3), axis=0))

    # Edge detection using prepared image
    edges_img = Image.fromarray((gray * MAX_PIXEL_VALUE).astype(np.uint8), mode='L')
    edges = np.array(edges_img.filter(ImageFilter.FIND_EDGES)) / MAX_PIXEL_VALUE
    edge_density = edges.mean()

    # Entropy calculation
    hist, _ = np.histogram(gray, bins=ENTROPY_BINS, range=(0, 1), density=True)
    bin_width = 1.0 / ENTROPY_BINS
    prob = hist * bin_width
    prob = prob[prob > ENTROPY_MIN_PROB]
    entropy = -np.sum(prob * np.log2(prob)) if len(prob) > 0 else 0.0

    # Texture (Sobel-like)
    gx = np.diff(gray, axis=1, prepend=0)
    gy = np.diff(gray, axis=0, prepend=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    texture_var = grad_mag.var() * 100.0

    # Autocorrelation (grid detection)
    h, w = gray.shape
    c = AUTO_CORR_PATCH_SIZE
    patch = gray[h // 2 - c : h // 2 + c, w // 2 - c : w // 2 + c]
    corr = np.fft.ifft2(np.abs(np.fft.fft2(patch)) ** 2)
    corr = np.fft.fftshift(corr.real)
    center_val = corr[c, c]
    neigh = np.concatenate(
        [
            corr[c - 1 : c, c - 1 : c],
            corr[c - 1 : c, c + 1 : c + 2],
            corr[c + 1 : c + 2, c - 1 : c],
            corr[c + 1 : c + 2, c + 1 : c + 2],
        ]
    )
    neigh_mean = neigh.mean()
    grid_ratio = center_val / (neigh_mean + ENTROPY_MIN_PROB)
    grid_penalty = max(0.0, (grid_ratio - GRID_RATIO_THRESHOLD)) * GRID_PENALTY_MULTIPLIER

    # Score assembly
    score = 0.0
    score += contrast * WEIGHT_CONTRAST
    score += sat_mean * WEIGHT_SATURATION
    score += entropy * WEIGHT_ENTROPY
    score += min(texture_var, TEXTURE_VAR_MAX) * WEIGHT_TEXTURE
    score += min(unique_colors / MAX_UNIQUE_COLORS, 1.0) * WEIGHT_COLORS

    # Penalties
    score -= (edge_density > EDGE_DENSITY_THRESHOLD) * PENALTY_HIGH_EDGE_DENSITY
    score -= (sat_std < SAT_STD_THRESHOLD) * PENALTY_LOW_SAT_STD
    score -= (contrast < CONTRAST_THRESHOLD) * PENALTY_LOW_CONTRAST
    score -= grid_penalty

    # Normalization
    norm = (score - SCORE_OFFSET) / SCORE_DIVISOR
    norm = np.clip(norm, 0.0, 1.0)

    if debug:
        print(f'contrast      : {contrast:.3f}')
        print(f'sat_mean/std  : {sat_mean:.3f}/{sat_std:.3f}')
        print(f'unique_colors : {unique_colors}')
        print(f'edge_density  : {edge_density:.3f}')
        print(f'entropy       : {entropy:.3f} (correct!)')
        print(f'texture_var   : {texture_var:.1f}')
        print(f'grid_penalty  : {grid_penalty:.3f}')
        print(f'raw_score     : {score:.2f} → norm {norm:.3f}')
    return float(norm)

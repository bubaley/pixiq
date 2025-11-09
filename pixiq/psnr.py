import numpy as np


def psnr(original: np.ndarray, compressed: np.ndarray, max_pixel: float = 255.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Used as a proxy for perceptual quality in compression algorithms.
    Higher PSNR generally correlates with better perceived image quality.
    """
    # Optimization: convert to float32 once and reuse
    orig_f32 = original.astype(np.float32, copy=False)
    comp_f32 = compressed.astype(np.float32, copy=False)
    mse = np.mean((orig_f32 - comp_f32) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel**2) / mse)

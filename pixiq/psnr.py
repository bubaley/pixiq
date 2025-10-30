# import pillow_avif  # noqa: F401
import numpy as np


def psnr(original: np.ndarray, compressed: np.ndarray, max_pixel=255.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Used as a proxy for perceptual quality in compression algorithms.
    Higher PSNR generally correlates with better perceived image quality.
    """
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    psnr_value = 10 * np.log10((max_pixel**2) / mse)
    return psnr_value

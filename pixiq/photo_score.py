import numpy as np
from PIL import Image, ImageFilter


def photo_score(image_array: np.ndarray, debug: bool = False) -> float:
    """Calculate photo score from numpy array (0.0 = UI-like, 1.0 = photo-like).

    Args:
        image_array: Image array (H, W, C) with values in range [0, 255]
        debug: If True, print debug information

    Returns:
        Photo score normalized to [0.0, 1.0]
    """
    # -------------------------------------------------
    # 1. Подготовка данных
    # -------------------------------------------------
    # Нормализуем массив и ресайзим до 256x256 для единообразия
    if len(image_array.shape) == 3:
        # RGB/RGBA
        img = Image.fromarray(image_array.astype(np.uint8))
        img = img.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        # Grayscale
        img = Image.fromarray(image_array.astype(np.uint8), mode='L')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.stack([arr, arr, arr], axis=2)  # Convert to RGB-like

    gray = arr.mean(axis=2)

    # -------------------------------------------------
    # 2. Признаки
    # -------------------------------------------------
    contrast = gray.std()

    sat_map = np.std(arr, axis=2)
    sat_mean = sat_map.mean()
    sat_std = sat_map.std()

    quant = np.round(arr * 31).astype(np.uint8)
    unique_colors = len(np.unique(quant.reshape(-1, 3), axis=0))

    # Вычисляем edges используя уже подготовленное изображение
    edges_img = Image.fromarray((gray * 255).astype(np.uint8), mode='L')
    edges = np.array(edges_img.filter(ImageFilter.FIND_EDGES)) / 255.0
    edge_density = edges.mean()

    # Исправленная энтропия
    hist, _ = np.histogram(gray, bins=64, range=(0, 1), density=True)
    bin_width = 1.0 / 64
    prob = hist * bin_width
    prob = prob[prob > 1e-12]
    entropy = -np.sum(prob * np.log2(prob)) if len(prob) > 0 else 0.0

    # Текстурность (Sobel-подобный)
    gx = np.diff(gray, axis=1, prepend=0)
    gy = np.diff(gray, axis=0, prepend=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    texture_var = grad_mag.var() * 100.0

    # Автокорреляция (проверка сетки)
    h, w = gray.shape
    c = 32
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
    grid_ratio = center_val / (neigh_mean + 1e-12)
    grid_penalty = max(0.0, (grid_ratio - 5.0)) * 0.04

    # -------------------------------------------------
    # 3. Сборка скора
    # -------------------------------------------------
    score = 0.0
    score += contrast * 2.8
    score += sat_mean * 2.2
    score += entropy * 0.8  # теперь entropy ~0–6 → вклад до 4.8
    score += min(texture_var, 150.0) * 0.006
    score += min(unique_colors / 3500.0, 1.0) * 0.9

    # Штрафы
    score -= (edge_density > 0.35) * 1.3
    score -= (sat_std < 0.03) * 1.1
    score -= (contrast < 0.08) * 1.6
    score -= grid_penalty

    # -------------------------------------------------
    # 4. Нормализация
    # -------------------------------------------------
    norm = (score - 1.0) / 7.0
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

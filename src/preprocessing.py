from pathlib import Path

import cv2
import numpy as np


def load_rgb_image(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise ValueError(f'could not read image: {path}')
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def crop_black_border(
        image: np.ndarray,
        threshold: int = 10,
        margin: float = 0.02
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold

    if not mask.any():
        return image
    
    y_indices, x_indices = mask.nonzero()

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    height, width = image.shape[:2]

    margin_y = int((y_max - y_min) * margin)
    margin_x = int((x_max - x_min) * margin)

    y_min = max(y_min - margin_y, 0)
    y_max = min(y_max + margin_y, height - 1)
    x_min = max(x_min - margin_x, 0)
    x_max = min(x_max + margin_x, width - 1)

    return image[y_min:y_max + 1, x_min:x_max + 1]


def resize_image(image: np.ndarray, size: int = 512) -> np.ndarray:
    resized = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )
    return resized 


def ben_graham_correction(image: np.ndarray, sigma: int = 30) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    corrected = cv2.addWeighted(image, 4, blurred, -4, 128)
    return corrected


def clahe_correction(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    l_channel_clahe = clahe.apply(l_channel)

    lab_clahe = cv2.merge([
        l_channel_clahe,
        a_channel,
        b_channel
    ])

    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return image_clahe


def green_ben_correction(image: np.ndarray, sigma: int = 30) -> np.ndarray:
    green_channel = image[:, :, 1]

    blurred = cv2.GaussianBlur(green_channel, (0, 0), sigma)
    corrected_green = cv2.addWeighted(green_channel, 4, blurred, -4, 128)

    green_ben_rgb = cv2.merge([
        corrected_green,
        corrected_green,
        corrected_green
    ])

    return green_ben_rgb


def preprocess_crop_resize(path: str | Path, size: int = 512) -> np.ndarray:
    image = load_rgb_image(path)
    image = crop_black_border(image)
    image = resize_image(image, size)
    return image


def preprocess_crop_ben_resize(path: str | Path, size: int = 512) -> np.ndarray:
    image = load_rgb_image(path)
    image = crop_black_border(image)
    image = ben_graham_correction(image)
    image = resize_image(image, size)
    return image


def preprocess_crop_clahe_resize(path: str | Path, size: int = 512) -> np.ndarray:
    image = load_rgb_image(path)
    image = crop_black_border(image)
    image = clahe_correction(image)
    image = resize_image(image, size)
    return image


def preprocess_crop_greenben_resize(path: str | Path, size: int = 512) -> np.ndarray:
    image = load_rgb_image(path)
    image = crop_black_border(image)
    image = green_ben_correction(image)
    image = resize_image(image, size)
    return image
"""Shared helpers for the cleaned-dataset cache.

The cache stores each fundus image cropped to the fundus and resized to a fixed
square size. Originals are never modified. Cached files keep the original
`<image>.jpeg` naming, so a cache folder can be used as a drop-in `images_dir`.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.preprocessing import crop_black_border

CACHE_EXT = ".jpeg"
DEFAULT_SIZE = 768
DEFAULT_JPEG_QUALITY = 95


def load_reduced(path: str | Path, target: int) -> tuple[np.ndarray, int, int]:
    """Decode a JPEG at the smallest DCT scale that is still >= target px.

    Returns the RGB image and the original (width, height).
    """
    with Image.open(path) as img:
        orig_w, orig_h = img.size
        img.draft("RGB", (target, target))
        image = np.asarray(img.convert("RGB"))
    return image, orig_w, orig_h


def make_cached_image(reduced_rgb: np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    cropped = crop_black_border(reduced_rgb)
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def save_rgb_jpeg(path: str | Path, image_rgb: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")  # OpenCV picks the format by extension
    ok = cv2.imwrite(str(tmp), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                     [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"could not write {path}")
    tmp.replace(path)  # atomic: an interrupted run never leaves a half-written image

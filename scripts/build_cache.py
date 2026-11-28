"""Build a downscaled image cache and per-image quality metrics in one pass.

Each source image is decoded once (at reduced resolution via JPEG DCT
scaling, which is much faster than a full decode), cropped to the fundus,
resized to a fixed square size and saved to the cache directory. In the same
pass we compute simple quality metrics used later for data cleaning.

Originals are never modified.

Example:
    python scripts/build_cache.py \
        --src ~/data/eyepacs/raw/train \
        --labels ~/data/eyepacs/raw/trainLabels.csv \
        --dst ~/data/eyepacs/cache_768 \
        --size 768
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import (  # noqa: E402
    CACHE_EXT,
    load_reduced,
    make_cached_image,
    save_rgb_jpeg,
)

MASK_THRESHOLD = 10
DARK_LEVEL = 20
BRIGHT_LEVEL = 245


def fundus_mask(image_rgb: np.ndarray, threshold: int = MASK_THRESHOLD) -> np.ndarray:
    """Largest connected bright region = the fundus disc."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > threshold).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def compute_metrics(cached_rgb: np.ndarray, reduced_rgb: np.ndarray) -> dict:
    h, w = reduced_rgb.shape[:2]
    full_mask = fundus_mask(reduced_rgb)
    mask_frac = float(full_mask.mean())

    if full_mask.any():
        ys, xs = np.nonzero(full_mask)
        bbox_w = (xs.max() - xs.min() + 1) / w
        bbox_h = (ys.max() - ys.min() + 1) / h
        bbox_aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    else:
        bbox_w = bbox_h = bbox_aspect = 0.0

    # Metrics below are computed on the cached (fixed-size) image so they are
    # comparable across different source resolutions.
    mask = fundus_mask(cached_rgb)
    gray = cv2.cvtColor(cached_rgb, cv2.COLOR_RGB2GRAY)
    green = cached_rgb[:, :, 1]

    if mask.sum() < 100:
        return {
            "mask_frac": mask_frac, "bbox_w": bbox_w, "bbox_h": bbox_h,
            "bbox_aspect": bbox_aspect, "cached_mask_frac": float(mask.mean()),
            "mean_brightness": np.nan, "contrast": np.nan,
            "frac_dark": np.nan, "frac_bright": np.nan,
            "sharpness": np.nan, "green_mean": np.nan,
            "red_mean": np.nan, "blue_mean": np.nan,
        }

    inner = cv2.erode(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    ).astype(bool)
    if inner.sum() < 100:
        inner = mask

    pixels = gray[mask]
    lap = cv2.Laplacian(green, cv2.CV_64F)

    return {
        "mask_frac": mask_frac,
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
        "bbox_aspect": float(bbox_aspect),
        "cached_mask_frac": float(mask.mean()),
        "mean_brightness": float(pixels.mean()),
        "contrast": float(pixels.std()),
        "frac_dark": float((pixels < DARK_LEVEL).mean()),
        "frac_bright": float((pixels > BRIGHT_LEVEL).mean()),
        "sharpness": float(lap[inner].var()),
        "red_mean": float(cached_rgb[:, :, 0][mask].mean()),
        "green_mean": float(green[mask].mean()),
        "blue_mean": float(cached_rgb[:, :, 2][mask].mean()),
    }


def process_one(args: tuple[str, Path, Path, int, int, bool]) -> dict:
    image_name, src_dir, dst_dir, size, jpeg_quality, overwrite = args
    src_path = src_dir / f"{image_name}.jpeg"
    dst_path = dst_dir / f"{image_name}{CACHE_EXT}"
    row = {"image": image_name, "readable": False, "error": ""}

    try:
        reduced, orig_w, orig_h = load_reduced(src_path, target=size)
        row.update(orig_w=orig_w, orig_h=orig_h,
                   decoded_w=reduced.shape[1], decoded_h=reduced.shape[0])

        cached = make_cached_image(reduced, size)
        if overwrite or not dst_path.exists():
            save_rgb_jpeg(dst_path, cached, jpeg_quality)

        row.update(compute_metrics(cached, reduced))
        row["phash"] = str(imagehash.phash(Image.fromarray(cached), hash_size=8))
        row["readable"] = True
    except Exception as exc:  # corrupted / truncated files end up here
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="folder with original .jpeg files")
    parser.add_argument("--labels", type=Path, default=None,
                        help="CSV with columns image,level (optional; default: all .jpeg in --src)")
    parser.add_argument("--dst", type=Path, required=True, help="output cache folder")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="process only first N images (testing)")
    parser.add_argument("--overwrite", action="store_true", help="re-write cached images that already exist")
    args = parser.parse_args()

    src_dir = args.src.expanduser()
    dst_dir = args.dst.expanduser()
    images_out = dst_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    if args.labels is not None:
        labels_df = pd.read_csv(args.labels.expanduser())
        names = labels_df["image"].tolist()
    else:
        labels_df = None
        names = sorted(p.stem for p in src_dir.glob("*.jpeg"))

    if args.limit:
        names = names[:args.limit]

    print(f"src:    {src_dir}\ndst:    {dst_dir}\nimages: {len(names)}\nsize:   {args.size}")

    tasks = [(n, src_dir, images_out, args.size, args.jpeg_quality, args.overwrite) for n in names]
    with Pool(args.workers) as pool:
        rows = list(tqdm(pool.imap_unordered(process_one, tasks, chunksize=16), total=len(tasks)))

    metrics_df = pd.DataFrame(rows)
    if labels_df is not None:
        metrics_df = labels_df.merge(metrics_df, on="image", how="left")
    metrics_df = metrics_df.sort_values("image").reset_index(drop=True)

    out_csv = dst_dir / "metrics.csv"
    metrics_df.to_csv(out_csv, index=False)

    n_bad = int((~metrics_df["readable"].astype(bool)).sum())
    print(f"\nsaved metrics: {out_csv}")
    print(f"unreadable images: {n_bad}")


if __name__ == "__main__":
    main()

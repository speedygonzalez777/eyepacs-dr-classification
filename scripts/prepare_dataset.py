"""Build the cleaned EyePACS dataset locally from the original Kaggle images.

Everyone downloads the original dataset themselves; this script then produces
exactly the same cleaned dataset on every machine, driven by
metadata/manifest.csv (which lives in git). Originals are only read, never
modified or deleted.

Output (--dst):
    images/<image>.jpeg   kept images only, cropped to the fundus and resized
                          to --size x --size (or the original files with --size 0)
    labels.csv            image, level, binary_label, patient_id, side, split
    dataset_info.json     manifest hash, size, counts (to check two copies match)

The script is incremental: images already present in --dst are skipped, and
images that the (updated) manifest no longer keeps are removed from --dst/images.
Re-run it after pulling a new manifest.

Usage with the training scripts: point IMAGES_DIR to <dst>/images and LABELS_CSV
to <dst>/labels.csv. Use the `split` column from labels.csv instead of
re-computing the split, otherwise the test set changes.

Defaults match scripts/download_dataset.py: originals in dataset/original/train,
output in dataset/clean_<size> (dataset/clean_original for --size 0), both
relative to the project root.

Example:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --src ~/data/eyepacs/raw/train --dst ~/data/eyepacs/clean_768
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import (  # noqa: E402
    CACHE_EXT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_SIZE,
    load_reduced,
    make_cached_image,
    save_rgb_jpeg,
)

DEFAULT_MANIFEST = PROJECT_ROOT / "metadata" / "manifest.csv"
DEFAULT_SRC = PROJECT_ROOT / "dataset" / "original" / "train"
SRC_EXT = ".jpeg"
LABEL_COLUMNS = ["image", "level", "binary_label", "patient_id", "side", "split"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard link (no extra disk space) when on the same drive, otherwise copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def process_one(task: tuple[Path, Path, int, int]) -> str:
    src_path, dst_path, size, jpeg_quality = task
    try:
        if size == 0:
            link_or_copy(src_path, dst_path)
        else:
            reduced, _, _ = load_reduced(src_path, target=size)
            save_rgb_jpeg(dst_path, make_cached_image(reduced, size), jpeg_quality)
        return ""
    except Exception as exc:
        return f"{src_path.name}: {type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="folder with the original Kaggle .jpeg files (default: dataset/original/train)")
    parser.add_argument("--dst", type=Path, default=None,
                        help="output folder for the cleaned dataset (default: dataset/clean_<size>)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help=f"output size in px (default {DEFAULT_SIZE}); 0 = keep original files")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--workers", type=int, default=None, help="default: all CPU cores")
    parser.add_argument("--no-prune", action="store_true",
                        help="do not remove images that the manifest no longer keeps")
    args = parser.parse_args(argv)
    if args.dst is None:
        args.dst = PROJECT_ROOT / "dataset" / (f"clean_{args.size}" if args.size else "clean_original")

    src_dir = args.src.expanduser().resolve()
    dst_dir = args.dst.expanduser().resolve()
    images_dir = dst_dir / "images"
    manifest_path = args.manifest.expanduser().resolve()

    if not src_dir.is_dir():
        sys.exit(f"--src does not exist: {src_dir}")
    if dst_dir == src_dir or src_dir.is_relative_to(dst_dir) or dst_dir.is_relative_to(src_dir):
        sys.exit("--dst must be a separate folder, not inside / containing --src (originals stay untouched)")

    info_path = dst_dir / "dataset_info.json"
    if info_path.exists():
        previous = json.loads(info_path.read_text())
        if previous.get("size") != args.size:
            sys.exit(f"{dst_dir} was built with --size {previous.get('size')}; "
                     f"use another --dst for --size {args.size}")

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    kept = manifest[manifest["keep"] == 1]

    missing = [name for name in kept["image"] if not (src_dir / f"{name}{SRC_EXT}").exists()]
    if missing:
        sys.exit(f"{len(missing)} images from the manifest are missing in --src, e.g. {missing[:5]}. "
                 "Is --src the folder with the original Kaggle train images?")

    images_dir.mkdir(parents=True, exist_ok=True)

    kept_names = set(kept["image"])
    if not args.no_prune:
        stale = [p for p in images_dir.glob(f"*{CACHE_EXT}") if p.stem not in kept_names]
        for path in stale:
            path.unlink()
        if stale:
            print(f"removed {len(stale)} images no longer kept by the manifest")
    for tmp in images_dir.glob(".*.tmp*"):  # leftovers from an interrupted run
        tmp.unlink()

    tasks = [
        (src_dir / f"{name}{SRC_EXT}", images_dir / f"{name}{CACHE_EXT}", args.size, args.jpeg_quality)
        for name in kept["image"]
        if not (images_dir / f"{name}{CACHE_EXT}").exists()
    ]

    print(f"manifest: {manifest_path}")
    print(f"src:      {src_dir}")
    print(f"dst:      {dst_dir}")
    print(f"kept:     {len(kept)} / {len(manifest)} images "
          f"({len(kept_names) - len(tasks)} already done, {len(tasks)} to process)")

    errors = []
    if tasks:
        with Pool(args.workers) as pool:
            for err in tqdm(pool.imap_unordered(process_one, tasks, chunksize=16), total=len(tasks)):
                if err:
                    errors.append(err)
    if errors:
        print(f"\n{len(errors)} images failed:")
        print("\n".join(errors[:20]))
        sys.exit(1)

    kept[LABEL_COLUMNS].to_csv(dst_dir / "labels.csv", index=False, lineterminator="\n")

    counts = kept.groupby(["split", "binary_label"]).size()
    info = {
        "manifest_sha256": sha256(manifest_path),
        "size": args.size,
        "jpeg_quality": args.jpeg_quality if args.size else None,
        "images": int(len(kept)),
        "excluded": int(len(manifest) - len(kept)),
        "counts": {f"{split}/label{label}": int(n) for (split, label), n in counts.items()},
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    print(f"\ndone: {len(kept)} images in {images_dir}")
    print(pd.crosstab(kept["split"], kept["binary_label"], margins=True))


if __name__ == "__main__":
    main()

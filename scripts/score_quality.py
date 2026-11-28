"""Score fundus image quality with QuickQual-MEME and save DenseNet121 features.

QuickQual (Engelmann et al., 2023): https://github.com/justinengelmann/QuickQual
  - ImageNet-pretrained DenseNet121 (torchvision IMAGENET1K_V1) as frozen feature extractor,
  - QuickQual-MEME: a 9-feature linear model fitted on EyeQ quality labels, giving p(bad).
The MEME weights below are copied verbatim from the QuickQual README
(commit a94feb02a79efa5380f6b0d863d1b80ab4c73e4e); nothing is downloaded except the
standard torchvision DenseNet121 weights.

Preprocessing follows QuickQual's image_preprocessing.py: crop black borders
(mean RGB > 15, +20 px buffer at full resolution), pad to square (aspect ratio
preserved), resize, then normalize with mean=std=0.5.

Full 1024-d features are saved too, so a classifier can be fitted on the
EyeQ manual labels later without re-running the network.

Example:
    python scripts/score_quality.py \
        --src ~/data/eyepacs/raw/train \
        --labels ~/data/eyepacs/raw/trainLabels.csv \
        --dst ~/data/eyepacs/quality
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import functional as F
from tqdm import tqdm

INPUT_SIZE = 512
BORDER_THRESHOLD = 15
BORDER_BUFFER_PX = 20  # at full resolution, as in QuickQual

MEME_FEATURES = [71, 109, 121, 53, 55, 123, 29, 133, 84]
MEME_WEIGHTS = torch.tensor([-1411.32, 517.09, 342.41, -707.9, 1442.09, -23.25, -541.64, -8.44, 5.44])
MEME_BIAS = torch.tensor([5.18])


def quickqual_preprocess(path: Path, size: int = INPUT_SIZE) -> Image.Image:
    with Image.open(path) as img:
        orig_w = img.size[0]
        # Fast reduced-resolution JPEG decode (DCT scaling); final size is still >= 2*size.
        img.draft("RGB", (size * 2, size * 2))
        img = img.convert("RGB")

    scale = img.size[0] / orig_w
    buffer = max(1, round(BORDER_BUFFER_PX * scale))

    arr = np.asarray(img)
    ys, xs = np.nonzero(arr.mean(-1) > BORDER_THRESHOLD)
    if len(xs) > 0:
        left, right = max(0, xs.min() - buffer), min(arr.shape[1], xs.max() + buffer)
        top, bottom = max(0, ys.min() - buffer), min(arr.shape[0], ys.max() + buffer)
        img = img.crop((left, top, right, bottom))

        width, height = img.size
        if width > height:
            pad = width - height
            img = F.pad(img, [0, pad // 2, 0, pad - pad // 2])
        else:
            pad = height - width
            img = F.pad(img, [pad // 2, 0, pad - pad // 2, 0])

    return img.resize((size, size), resample=Image.LANCZOS)


class QualityDataset(Dataset):
    def __init__(self, names: list[str], src_dir: Path):
        self.names = names
        self.src_dir = src_dir

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = quickqual_preprocess(self.src_dir / f"{self.names[idx]}.jpeg")
        tensor = F.normalize(F.to_tensor(img), [0.5] * 3, [0.5] * 3)
        return tensor, idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="folder with original .jpeg files")
    parser.add_argument("--labels", type=Path, required=True, help="CSV with column 'image'")
    parser.add_argument("--dst", type=Path, required=True, help="output folder")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--limit", type=int, default=None, help="process only first N images (testing)")
    args = parser.parse_args()

    src_dir = args.src.expanduser()
    dst_dir = args.dst.expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)

    names = pd.read_csv(args.labels.expanduser())["image"].tolist()
    if args.limit:
        names = names[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)

    loader = DataLoader(QualityDataset(names, src_dir), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False)

    features = np.zeros((len(names), 1024), dtype=np.float32)
    with torch.inference_mode():
        for batch, idx in tqdm(loader, desc=f"scoring on {device}"):
            features[idx.numpy()] = model(batch.to(device)).cpu().numpy()

    feats = torch.from_numpy(features)
    p_bad = torch.sigmoid(feats[:, MEME_FEATURES] @ MEME_WEIGHTS + MEME_BIAS).numpy()

    pd.DataFrame({"image": names, "quickqual_meme_p_bad": p_bad}).to_csv(
        dst_dir / "quickqual_meme.csv", index=False)
    np.save(dst_dir / "densenet121_features.npy", features.astype(np.float16))
    pd.Series(names, name="image").to_csv(dst_dir / "densenet121_features_index.csv", index=False)

    print(f"saved to {dst_dir}")


if __name__ == "__main__":
    main()

# EyePACS DR Classification

Binary diabetic retinopathy classification on EyePACS fundus images:

- `0 = no DR`
- `1 = DR`

The current training pipeline uses a patient-level split, a ConvNeXt-Large transfer-learning model, and selectable preprocessing variants for comparing image preparation strategies.

## Project Structure

- `scripts/train_transfer_learning.py` - main transfer-learning training script.
- `src/dataset.py` - EyePACS dataset loader, ImageNet normalization, optional train-only augmentations.
- `src/preprocessing.py` - image loading and preprocessing variants.
- `src/splits.py` - patient-level train/validation/test split.
- `src/transfer_learning_model.py` - ConvNeXt-Large binary classifier.
- `results/transfer_learning/<preprocessing>/` - local training outputs.

## Environment

Recommended:

- Windows or Linux with Python 3.12-compatible PyTorch environment.
- NVIDIA GPU with CUDA for practical ConvNeXt-Large training.
- Enough disk space for EyePACS images and local training outputs.

Create and activate a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check PyTorch and CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dataset Paths

The training script currently expects:

```python
LABELS_CSV = Path("F:/ZPB/dataset/original/trainLabels.csv")
IMAGES_DIR = Path("F:/ZPB/dataset/original/train")
```

If your dataset is stored elsewhere, update these constants in `scripts/train_transfer_learning.py` before starting training. Image files are expected to match the EyePACS naming from `trainLabels.csv`, with `.jpeg` appended by the dataset loader.

## Preprocessing Variants

Choose one preprocessing variant by editing the `PREPROCESSING` constant in `scripts/train_transfer_learning.py`:

```python
PREPROCESSING = "crop_resize"
PREPROCESSING = "crop_ben_resize"
PREPROCESSING = "crop_clahe_resize"
PREPROCESSING = "crop_greenben_resize"
```

Each run writes to a separate folder:

```text
results/transfer_learning/<preprocessing>/
```

For example, `PREPROCESSING = "crop_ben_resize"` writes to:

```text
results/transfer_learning/crop_ben_resize/
```

## Training

Default training settings:

- Model: ConvNeXt-Large pretrained on ImageNet.
- Image size: `512`.
- Batch size: `8`.
- Epochs: `30`.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: `AdamW`.
- Split: patient-level train/validation/test split from `src/splits.py`.

Start transfer-learning training:

```powershell
python scripts/train_transfer_learning.py
```

Do not start another run in the same output folder while an existing training run is still writing results.

## TensorBoard

TensorBoard logs are written under each preprocessing output directory:

```text
results/transfer_learning/<preprocessing>/tensorboard/
```

Start TensorBoard from the project root:

```powershell
tensorboard --logdir results/transfer_learning
```

Then open the URL printed by TensorBoard in your browser.

## Outputs

Each preprocessing run saves:

- `best_model.pt` - best checkpoint by validation ROC-AUC.
- `history.csv` - per-epoch train/validation loss and validation metrics.
- `test_results.csv` - final test metrics after loading the best checkpoint.
- `tensorboard/` - TensorBoard event logs.

These outputs are local experiment artifacts and should not be committed.

## Running Another Preprocessing Variant

To compare another preprocessing method:

1. Wait for the current training run to finish, or use a separate machine/environment.
2. Change `PREPROCESSING` in `scripts/train_transfer_learning.py`.
3. Run `python scripts/train_transfer_learning.py`.
4. Compare `history.csv`, `test_results.csv`, and TensorBoard curves across `results/transfer_learning/<preprocessing>/`.

## Augmentations

Random augmentations are enabled only for the training dataset. Validation and test datasets do not use random augmentations, so metrics are evaluated on deterministic preprocessed images.

Current train-only augmentations are intentionally mild for fundus images:

- random horizontal flip,
- small rotation around +/- 15 degrees,
- light brightness and contrast changes,
- very small optional zoom/crop.

The pipeline intentionally avoids aggressive blur, strong color shifts, large deformations, and random erasing/cutout because these can hide small disease-related lesions.

## Git Hygiene

Do not commit local datasets or experiment artifacts. Keep these ignored:

- `results/`
- checkpoint files such as `*.pt`, `*.pth`, `*.ckpt`
- TensorBoard logs such as `runs/` and `results/**/tensorboard/`
- local virtual environments such as `.venv/`
- Python caches such as `__pycache__/` and `.pytest_cache/`
- large local dataset directories such as `dataset/`, `data/`, and `datasets/`


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

## Cleaned Dataset

The cleaned dataset is not uploaded anywhere. Everyone builds it locally from the original Kaggle images ([dreamer07/eyepacs](https://www.kaggle.com/datasets/dreamer07/eyepacs), 35,126 train images) using the manifest stored in git:

- `metadata/manifest.csv` - one row per original image: `level`, `binary_label`, `patient_id`, fixed `split`, `eyeq_quality`, `p_reject`, `manual_decision`, `keep`, `exclude_reason`.
- `metadata/eyeq/` - source, license and citation of the EyeQ quality labels used by the manifest.

Current manifest (`v2-eyeq-model-manual`) keeps 30,484 of 35,126 images (22,223 no DR, 8,261 DR) and excludes 4,642:

| `exclude_reason` | rule | images |
|---|---|---|
| `eyeq_reject` | graded **Reject** by EyeQ (12,543 images have an EyeQ grade) | 2,320 |
| `model_reject` | no EyeQ grade, quality model `p_reject >= 0.5`, DR level 0-1 | 2,195 |
| `manual_reject` | no EyeQ grade, `p_reject >= 0.5`, DR level 2-4, rejected in manual review (842 reviewed, 715 kept) | 127 |

The quality model is a logistic regression on DenseNet121 features from QuickQual, trained on the EyeQ grades (cross-validated ROC-AUC 0.99 for Reject vs rest). Flagged images with DR level 2-4 are reviewed by hand, because the model tends to flag laser scars and severe pathology as artefacts. The scripts, notebooks and manual decisions used to build the manifest are on the `users/rmichal/data-cleaning` branch; to train you only need the manifest and `scripts/prepare_dataset.py`.

Build the cleaned dataset (originals are only read, never modified):

```powershell
python scripts/prepare_dataset.py
```

By default it reads the originals from `dataset/original/train` and writes to `dataset/clean_768` (both relative to the project root). Other locations can be passed with `--src` and `--dst`, for example `--src F:/ZPB/dataset/original/train --dst F:/ZPB/dataset/clean_768`.

This writes:

- `clean_768/images/<image>.jpeg` - kept images, cropped to the fundus and resized to 768x768 (about 3.5 GB),
- `clean_768/labels.csv` - `image, level, binary_label, patient_id, side, split`,
- `clean_768/dataset_info.json` - manifest hash and counts; compare it between machines to confirm you have the same dataset.

Use `--size 0` to keep the original full-resolution files instead (hard links when on the same drive, otherwise copies). The script is incremental: after pulling an updated manifest, run it again and it only adds or removes the changed images.

To train on it, set `IMAGES_DIR` to `dataset/clean_768/images` and `LABELS_CSV` to `dataset/clean_768/labels.csv`, and **use the `split` column from `labels.csv` instead of calling `create_patient_level_split` again**. Re-splitting the filtered set would change the test set and make results incomparable. The stored split is identical to the one `create_patient_level_split` produces on the full dataset (seed 42).

Note: preprocessing variants (`crop_ben_resize`, `crop_clahe_resize`, ...) applied to 768 px cached images are not identical to the same variants applied to full-resolution originals, because filters such as the Ben Graham blur (`sigma=30`) act relative to image size. Compare experiments only within the same image source.

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


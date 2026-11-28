# EyePACS DR Classification

Binary diabetic retinopathy classification on EyePACS fundus images:

- `0 = no DR`
- `1 = DR`

The current training pipeline uses a patient-level split, selectable transfer-learning models, and selectable preprocessing variants for comparing image preparation strategies.

## Project Structure

- `scripts/train_transfer_learning.py` - main transfer-learning training script.
- `src/dataset.py` - EyePACS dataset loader, ImageNet normalization, optional train-only augmentations.
- `src/preprocessing.py` - image loading and preprocessing variants.
- `src/splits.py` - patient-level train/validation/test split.
- `src/transfer_learning_model.py` - transfer-learning model factories.
- `results/transfer_learning/<model>_<preprocessing>/` - local training outputs.

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

The training script resolves the dataset relative to the project directory:

```python
DATASET_ROOT = PROJECT_ROOT / "dataset" / "original"
LABELS_CSV = DATASET_ROOT / "trainLabels.csv"
IMAGES_DIR = DATASET_ROOT / "train"
```

Place the data in:

```text
dataset/original/trainLabels.csv
dataset/original/train/
```

Download the Kaggle dataset into this layout from the project root:

```powershell
python -m pip install kagglehub
python scripts/download_dataset.py
```

KaggleHub downloads directly to `dataset/kagglehub_download` on the project
drive. The script then moves the labels and JPEG images (including files inside
ZIP archives) into `dataset/original`, which is the location used by the
training scripts. If `dataset/original` is already complete, the script exits
without downloading again. Kaggle authentication and acceptance of the
dataset's terms may be required.

To use a dataset stored elsewhere, update `DATASET_ROOT` in `scripts/train_transfer_learning.py`. Image files are expected to match the EyePACS naming from `trainLabels.csv`, with `.jpeg` appended by the dataset loader.

## Transfer-Learning Models

Set `model_name` in `DEFAULT_CONFIG` in `scripts/train_transfer_learning.py` to one of the supported values:

```python
model_name="convnext_large"
model_name="convnext_base"
model_name="swin_v2_tiny"
```

Available models:

- ConvNeXt-Large: large ImageNet-pretrained baseline.
- ConvNeXt-Base: smaller ConvNeXt variant for comparison.
- SwinV2-Tiny: smaller modern transformer architecture.

Full fine-tuning is the default. The script also supports partial fine-tuning for ConvNeXt-Large and the ConvNeXt-Base experiments below.

## Preprocessing Variants

Set `preprocessing` in `DEFAULT_CONFIG` to one of the supported values:

```python
preprocessing="crop_resize"
preprocessing="crop_ben_resize"
preprocessing="crop_clahe_resize"
preprocessing="crop_greenben_resize"
```

The original entry point writes under `results/transfer_learning/` using `RUN_NAME`. Choose a fresh `RUN_NAME` when changing the default experiment. The script refuses to use an existing run directory.

## Training

Default training settings when no command-line option is supplied:

- Model: ConvNeXt-Large pretrained on ImageNet.
- Preprocessing: `crop_clahe_resize`.
- Image size: `512`.
- Batch size: configured in `scripts/train_transfer_learning.py`.
- Epochs: `30`.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: `AdamW`.
- Split: patient-level train/validation/test split from `src/splits.py`.

Start transfer-learning training:

```powershell
python scripts/train_transfer_learning.py
```

## ConvNeXt-Base overfitting experiments

From the project root, run each variant separately:

```powershell
python scripts/train_transfer_learning.py --experiment base_baseline
python scripts/train_transfer_learning.py --experiment base_scheduler
python scripts/train_transfer_learning.py --experiment base_unfreeze
python scripts/train_transfer_learning.py --experiment base_scheduler_unfreeze
```

All four use ImageNet weights, CLAHE at 512 × 512, the same patient split, seeds and DataLoader settings, the same train augmentations, batch size 15, and AdamW with initial LR `1e-4` and weight decay `1e-4`. The classifier has no extra dropout.

- `base_baseline` reproduces the historical Base settings with full fine-tuning from epoch 1 and a constant LR.
- `base_scheduler` fine-tunes all layers from epoch 1 and halves LR after three epochs without a validation ROC-AUC improvement greater than `0.001`. The scheduler uses a one-epoch cooldown and a minimum LR of `1e-6`.
- `base_unfreeze` uses a constant LR and trains the classifier in epochs 1–2, adds stage 4 in epochs 3–5, adds stage 3 in epochs 6–8, then trains the entire model from epoch 9.
- `base_scheduler_unfreeze` uses both techniques with the same settings.

Every epoch is eligible for the best checkpoint by validation ROC-AUC. Early stopping counts eight epochs without improvement over that best score, starting with full fine-tuning. For gradual unfreezing, the first epoch that can count toward early stopping is epoch 9.

Each run creates `results/transfer_learning/<variant>/<run_id>/` with its effective `config.json`, `split.csv`, best checkpoint by validation ROC-AUC, `history.csv`, and TensorBoard logs. The history includes the LR, the number of trainable parameters, and whether early stopping is active for each epoch. Compare validation ROC-AUC and the validation loss curves to the historical Base run in `convnext_base_crop_clahe_resize`.

Once a run is selected on validation, evaluate its best checkpoint on the saved test split:

```powershell
python scripts/train_transfer_learning.py --evaluate-test results/transfer_learning/base_scheduler/<run_id>
```

Replace `<run_id>` with the actual directory name and use the selected variant's path. This writes `test_results.csv` with the checkpoint epoch and run ID.

## EfficientNet-B3 partial fine-tuning

Run from the project root:

```powershell
python scripts/train_transfer_learning.py --experiment efficientnet_b3_partial
```

The model starts with ImageNet weights and uses the existing patient-level split, 512×512 CLAHE preprocessing, train-only augmentations, and binary DR metrics. Training uses classifier dropout `0.3`, batch size 15, AdamW, CUDA FP16 AMP, and cuDNN benchmarking.

Epochs 1 and 2 train the classifier at `1e-4`. Starting at epoch 3, the last individual MBConv block and final convolution also train at `1e-5`. Frozen layers stay in evaluation mode so their BatchNorm statistics stay fixed. Every epoch can produce the best checkpoint by validation ROC-AUC. Early stopping counts four epochs without improvement, starting at epoch 3.

Each run writes `config.json`, `split.csv`, `best_model.pt`, `history.csv`, and TensorBoard logs under `results/transfer_learning/efficientnet_b3_partial/<run_id>/`. Use `--evaluate-test RUN_DIR` with a selected run directory to evaluate its best checkpoint on the saved test split.

## EfficientNet-B3 full fine-tuning

Run from the project root:

```powershell
python scripts/train_transfer_learning.py --experiment efficientnet_b3_full
```

All backbone layers train from epoch 1 at `1e-5` and the classifier trains at `1e-4`. The experiment shares the partial variant's ImageNet initialization, data pipeline, classifier dropout, batch size and AdamW weight decay. Early stopping monitors validation ROC-AUC from epoch 1 with patience 4, up to 30 epochs. Learning rates are constant. Each run saves its configuration, split, best checkpoint, history and TensorBoard logs under `results/transfer_learning/efficientnet_b3_full/<run_id>/`.

Evaluation now computes sigmoid probabilities in FP32. Earlier CUDA AMP runs computed the sigmoid in FP16, which could introduce extra ties in ROC-AUC and average precision.

The full B3 experiment also runs the validation and test forward pass in FP32. In a replay of epoch 1, the expansion convolution in the final MBConv stage exceeded the FP16 range for `669_right.jpeg` before the sigmoid. Training uses AMP FP16. Numerical failures are recorded in the run's `numerics.jsonl` with one-based epoch and batch numbers, the stage, and image filenames. Backward overflows handled by GradScaler are recorded separately from fatal forward or model-state errors.

## ConvNeXt-Tiny regularized

Run `python scripts/train_transfer_learning.py --experiment convnext_tiny_regularized` from the project root. The run fine-tunes all layers with CUDA BF16, a classifier dropout of 0.3 and constant AdamW learning rates of `1e-5` for the backbone and `1e-4` for the classifier.

Each new minimum validation BCE saves a checkpoint. Early stopping uses patience 4 and a minimum improvement of 0.001. Each epoch also records `train_eval_loss` in evaluation mode on the same 1024 unaugmented training images, sampled proportionally by DR grade with seed 42. The image list is saved as `train_eval.csv` under `results/transfer_learning/convnext_tiny_regularized/<run_id>/`.

## ConvNeXt-Tiny partial fine-tuning

Run `python scripts/train_transfer_learning.py --experiment convnext_tiny_partial` from the project root. This run starts from ImageNet weights and trains only `features[7]` and the classifier. Earlier stages stay in evaluation mode during training, disabling their stochastic depth. Data, optimizer settings, BF16, checkpoint selection and early stopping match `convnext_tiny_regularized`.

Each run under `results/transfer_learning/convnext_tiny_partial/<run_id>/` records the trainable parameter count and `val_minus_train_eval_loss` in its history and TensorBoard logs. A positive gap means validation loss is higher than evaluation-mode loss on the fixed 1024-image training sample.

## ConvNeXt-Tiny five-grade training

Run `python scripts/train_transfer_learning.py --experiment convnext_tiny_5class` from the project root. Each run copies the saved patient split from `convnext_tiny_regularized/20261123T081734Z-349d2677` and fine-tunes ImageNet weights using unweighted five-class cross-entropy on grades 0–4. Validation uses the existing binary metrics at threshold 0.5, with the DR score given by the summed softmax probabilities of grades 1–4.

`train_5class_ce` and `val_5class_ce` measure grade prediction. The checkpoint and early stopping monitor `val_binary_bce`, computed in FP32 from the aggregated binary logit. History, TensorBoard events, the copied split, configuration and best checkpoint are stored under `results/transfer_learning/convnext_tiny_5class/<run_id>/`.

To compare preprocessing, run `python scripts/train_transfer_learning.py --experiment convnext_tiny_5class_no_clahe`. This variant uses `crop_resize` at 512×512 and writes to `results/transfer_learning/convnext_tiny_5class_no_clahe/<run_id>/`. Both five-grade variants display epoch metrics to four decimal places and retain full precision in the saved results.

## TensorBoard

TensorBoard logs are written under each run output directory:

```text
results/transfer_learning/<model>_<preprocessing>/tensorboard/
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
- `test_results.csv` - final test metrics after loading the best checkpoint. The Base experiments generate this file through `--evaluate-test`.
- `tensorboard/` - TensorBoard event logs.

These outputs are local experiment artifacts and should not be committed.

## Running Another Experiment

To compare another model or preprocessing method:

1. Wait for the current training run to finish, or use a separate machine/environment.
2. Change `model_name` and/or `preprocessing` in `DEFAULT_CONFIG` in `scripts/train_transfer_learning.py` and choose a fresh `RUN_NAME`.
3. Run `python scripts/train_transfer_learning.py`.
4. Compare `history.csv`, `test_results.csv`, and TensorBoard curves across `results/transfer_learning/<model>_<preprocessing>/`.

## DINOv3 Linear Probe

The separate DINOv3 experiment uses `facebook/dinov3-vits16-pretrain-lvd1689m` as a completely frozen ViT-S/16 backbone. Images are passed through the backbone once, and only a binary linear classifier is trained on the cached embeddings.

This experiment does not modify or fine-tune the DINOv3 backbone. All backbone parameters have `requires_grad=False`, backbone inference runs under `torch.no_grad()`, and only `nn.Linear(hidden_size, 1)` is trainable.

The pipeline uses deterministic `crop_clahe_resize` preprocessing at `512x512` for train, validation, and test. It does not use random augmentations and uses the DINOv3 `AutoImageProcessor` directly, without the ImageNet normalization from `EyePACSDataset`.

Run the required lightweight smoke test:

```powershell
python scripts/train_dinov3_linear_probe.py --smoke-test-only
```

The smoke test loads one or two images and verifies input, embedding, and logits shapes as well as the frozen-backbone configuration. It does not perform full embedding extraction or training.

Start the full experiment:

```powershell
python scripts/train_dinov3_linear_probe.py
```

Embeddings and results are stored under:

```text
results/dinov3_linear_probe/vits16_crop_clahe_resize/
```

The directory contains:

- `embeddings/train.pt`
- `embeddings/val.pt`
- `embeddings/test.pt`
- `best_model.pt`
- `history.csv`
- `test_results.csv`
- `tensorboard/`

Existing embedding cache files are reused automatically. Force a new extraction with:

```powershell
python scripts/train_dinov3_linear_probe.py --recompute-embeddings
```

Compare the DINOv3 linear-probe metrics with the fully fine-tuned ConvNeXt and SwinV2 runs. This comparison shows how useful frozen foundation-model embeddings are relative to end-to-end transfer learning.

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


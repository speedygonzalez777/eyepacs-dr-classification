from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoImageProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dinov3_linear_probe_model import (
    DINOv3LinearProbe,
    MODEL_NAME,
    print_parameter_summary,
)
from src.preprocessing import preprocess_crop_clahe_resize
from src.splits import create_patient_level_split


LABELS_CSV = Path("F:/ZPB/dataset/original/trainLabels.csv")
IMAGES_DIR = Path("F:/ZPB/dataset/original/train")

PREPROCESSING = "crop_clahe_resize"
IMAGE_SIZE = 512
EMBEDDING_BATCH_SIZE = 4
BATCH_SIZE = 256
NUM_EPOCHS = 100
LINEAR_PROBE_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 20
RANDOM_SEED = 42

NUM_WORKERS = 2
PIN_MEMORY = True

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "dinov3_linear_probe"
    / "vits16_crop_clahe_resize"
)
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DINOv3ImageDataset(Dataset):
    def __init__(
        self,
        labels_df: pd.DataFrame,
        images_dir: str | Path,
        processor: AutoImageProcessor,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.labels_df = labels_df.reset_index(drop=True).copy()
        if "binary_label" not in self.labels_df.columns:
            self.labels_df["binary_label"] = (
                self.labels_df["level"] > 0
            ).astype(int)

        self.images_dir = Path(images_dir)
        self.processor = processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels_df.iloc[idx]
        image_path = self.images_dir / f"{row['image']}.jpeg"
        image = preprocess_crop_clahe_resize(image_path, size=self.image_size)

        processed = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = processed["pixel_values"].squeeze(0)

        expected_shape = (3, self.image_size, self.image_size)
        if tuple(pixel_values.shape) != expected_shape:
            raise ValueError(
                f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}. "
                f"Expected {expected_shape}."
            )

        label = torch.tensor(row["binary_label"], dtype=torch.float32)
        return pixel_values, label


def create_processor() -> AutoImageProcessor:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    processor.do_resize = False
    if hasattr(processor, "do_center_crop"):
        processor.do_center_crop = False
    return processor


def create_split_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df = create_patient_level_split(
        labels_df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_seed=RANDOM_SEED,
    )

    train_df = labels_df[labels_df["split"] == "train"].copy()
    val_df = labels_df[labels_df["split"] == "val"].copy()
    test_df = labels_df[labels_df["split"] == "test"].copy()

    print("Train samples:", len(train_df))
    print("Val samples:", len(val_df))
    print("Test samples:", len(test_df))

    return train_df, val_df, test_df


def create_image_datasets(
    processor: AutoImageProcessor,
) -> dict[str, DINOv3ImageDataset]:
    train_df, val_df, test_df = create_split_dataframes()
    return {
        "train": DINOv3ImageDataset(train_df, IMAGES_DIR, processor),
        "val": DINOv3ImageDataset(val_df, IMAGES_DIR, processor),
        "test": DINOv3ImageDataset(test_df, IMAGES_DIR, processor),
    }


def smoke_test(
    model: DINOv3LinearProbe,
    dataset: DINOv3ImageDataset,
    device: torch.device,
    sample_count: int = 2,
) -> None:
    sample_count = min(sample_count, len(dataset))
    samples = [dataset[index][0] for index in range(sample_count)]
    pixel_values = torch.stack(samples).to(device)

    expected_input_shape = (sample_count, 3, IMAGE_SIZE, IMAGE_SIZE)
    if tuple(pixel_values.shape) != expected_input_shape:
        raise ValueError(
            f"Unexpected smoke-test input shape: {tuple(pixel_values.shape)}. "
            f"Expected {expected_input_shape}."
        )

    model.eval()
    with torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda",
    ):
        embeddings = model.extract_embeddings(pixel_values)
        logits = model.classifier(embeddings)

    expected_embedding_shape = (sample_count, model.hidden_size)
    expected_logits_shape = (sample_count, 1)

    if tuple(embeddings.shape) != expected_embedding_shape:
        raise ValueError(
            f"Unexpected embedding shape: {tuple(embeddings.shape)}. "
            f"Expected {expected_embedding_shape}."
        )
    if tuple(logits.shape) != expected_logits_shape:
        raise ValueError(
            f"Unexpected logits shape: {tuple(logits.shape)}. "
            f"Expected {expected_logits_shape}."
        )

    summary = model.parameter_summary()
    if not summary["backbone_frozen"]:
        raise ValueError("DINOv3 backbone is not fully frozen.")
    if summary["trainable_parameters"] != summary["classifier_parameters"]:
        raise ValueError("Parameters outside the classifier are trainable.")

    model.train()
    if model.backbone.training:
        raise ValueError("DINOv3 backbone entered training mode.")
    model.eval()

    print("Smoke test input shape:", tuple(pixel_values.shape))
    print("Smoke test embedding shape:", tuple(embeddings.shape))
    print("Smoke test logits shape:", tuple(logits.shape))
    print_parameter_summary(model)


def cache_path(split_name: str) -> Path:
    return EMBEDDINGS_DIR / f"{split_name}.pt"


def cache_is_complete() -> bool:
    return all(cache_path(split).exists() for split in ("train", "val", "test"))


def extract_embeddings(
    model: DINOv3LinearProbe,
    dataset: DINOv3ImageDataset,
    split_name: str,
    device: torch.device,
) -> dict[str, torch.Tensor | int | str]:
    loader = DataLoader(
        dataset,
        batch_size=EMBEDDING_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    embeddings = []
    labels = []
    model.eval()

    for pixel_values, batch_labels in tqdm(
        loader,
        desc=f"Extracting {split_name} embeddings",
        leave=False,
    ):
        pixel_values = pixel_values.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            batch_embeddings = model.extract_embeddings(pixel_values)

        embeddings.append(batch_embeddings.detach().float().cpu())
        labels.append(batch_labels.float().cpu())

    return {
        "embeddings": torch.cat(embeddings),
        "labels": torch.cat(labels),
        "hidden_size": model.hidden_size,
        "model_name": MODEL_NAME,
        "preprocessing": PREPROCESSING,
        "image_size": IMAGE_SIZE,
    }


def build_embedding_cache(
    datasets: dict[str, DINOv3ImageDataset],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor | int | str]]:
    model = DINOv3LinearProbe().to(device)
    print_parameter_summary(model)
    smoke_test(model, datasets["train"], device)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    for split_name, dataset in datasets.items():
        split_cache = extract_embeddings(model, dataset, split_name, device)
        torch.save(split_cache, cache_path(split_name))
        cache[split_name] = split_cache

    return cache


def load_embedding_cache() -> dict[str, dict[str, torch.Tensor | int | str]]:
    print("Using embedding cache:", EMBEDDINGS_DIR)
    return {
        split: torch.load(cache_path(split), map_location="cpu")
        for split in ("train", "val", "test")
    }


def create_embedding_loaders(
    cache: dict[str, dict[str, torch.Tensor | int | str]],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loaders = []
    for split_name in ("train", "val", "test"):
        split_cache = cache[split_name]
        dataset = TensorDataset(
            split_cache["embeddings"].float(),
            split_cache["labels"].float(),
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=split_name == "train",
                num_workers=0,
                pin_memory=PIN_MEMORY,
            )
        )
    return tuple(loaders)


def calculate_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)

    if len(np.unique(labels)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = roc_auc_score(labels, probabilities)
        pr_auc = average_precision_score(labels, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(labels, predictions, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_one_epoch(
    classifier: nn.Linear,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    classifier.train()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(data_loader, desc="Training linear probe", leave=False)
    for embeddings, labels in progress:
        embeddings = embeddings.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = classifier(embeddings).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = embeddings.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(train_loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


def evaluate(
    classifier: nn.Linear,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    classifier.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all = []
    probabilities_all = []

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = classifier(embeddings).squeeze(1)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

            batch_size = embeddings.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())

    labels_np = np.concatenate(labels_all)
    probabilities_np = np.concatenate(probabilities_all)
    metrics = calculate_binary_metrics(labels_np, probabilities_np)
    return total_loss / total_samples, metrics


def print_metrics(prefix: str, loss: float, metrics: dict[str, float]) -> None:
    print(
        f"{prefix} loss: {loss:.4f} | "
        f"roc_auc: {metrics['roc_auc']:.4f} | "
        f"pr_auc: {metrics['pr_auc']:.4f} | "
        f"balanced_acc: {metrics['balanced_accuracy']:.4f} | "
        f"precision: {metrics['precision']:.4f} | "
        f"recall: {metrics['recall']:.4f} | "
        f"specificity: {metrics['specificity']:.4f} | "
        f"f1: {metrics['f1']:.4f}"
    )


def log_tensorboard(
    writer: SummaryWriter,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, float],
) -> None:
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    for metric_name in (
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    ):
        writer.add_scalar(f"Val/{metric_name}", metrics[metric_name], epoch)
    writer.flush()


def train_classifier(
    cache: dict[str, dict[str, torch.Tensor | int | str]],
    device: torch.device,
) -> None:
    train_loader, val_loader, test_loader = create_embedding_loaders(cache)
    hidden_size = int(cache["train"]["hidden_size"])

    classifier = nn.Linear(hidden_size, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LINEAR_PROBE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_model_path = OUTPUT_DIR / "best_model.pt"
    history_path = OUTPUT_DIR / "history.csv"
    writer = SummaryWriter(log_dir=OUTPUT_DIR / "tensorboard")
    best_val_roc_auc = -1.0
    epochs_without_improvement = 0
    history = []

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
            train_loss = train_one_epoch(
                classifier,
                train_loader,
                criterion,
                optimizer,
                device,
            )
            val_loss, val_metrics = evaluate(
                classifier,
                val_loader,
                criterion,
                device,
            )

            print(f"Train loss: {train_loss:.4f}")
            print_metrics("Val", val_loss, val_metrics)
            log_tensorboard(writer, epoch, train_loss, val_loss, val_metrics)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{
                        f"val_{key}": value
                        for key, value in val_metrics.items()
                    },
                }
            )
            pd.DataFrame(history).to_csv(history_path, index=False)

            val_roc_auc = val_metrics["roc_auc"]
            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "classifier_state_dict": classifier.state_dict(),
                        "best_val_roc_auc": best_val_roc_auc,
                        "hidden_size": hidden_size,
                        "model_name": MODEL_NAME,
                        "preprocessing": PREPROCESSING,
                    },
                    best_model_path,
                )
                print("Saved new best model:", best_model_path)
            else:
                epochs_without_improvement += 1
                print(
                    f"No improvement for {epochs_without_improvement}/"
                    f"{EARLY_STOPPING_PATIENCE} epochs"
                )

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
    finally:
        writer.close()

    checkpoint = torch.load(best_model_path, map_location=device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    test_loss, test_metrics = evaluate(
        classifier,
        test_loader,
        criterion,
        device,
    )
    print_metrics("Test", test_loss, test_metrics)

    pd.DataFrame(
        [
            {
                "test_loss": test_loss,
                **{
                    f"test_{key}": value
                    for key, value in test_metrics.items()
                },
            }
        ]
    ).to_csv(OUTPUT_DIR / "test_results.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear probe on frozen DINOv3 embeddings."
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Recompute train, validation, and test embedding cache.",
    )
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="Run input, embedding, logits, and frozen-backbone checks only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("Preprocessing:", PREPROCESSING)
    print("Image size:", IMAGE_SIZE)
    print("Output directory:", OUTPUT_DIR)

    processor = create_processor()
    datasets = create_image_datasets(processor)

    if args.smoke_test_only:
        model = DINOv3LinearProbe().to(device)
        smoke_test(model, datasets["train"], device)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if cache_is_complete() and not args.recompute_embeddings:
        cache = load_embedding_cache()
    else:
        cache = build_embedding_cache(datasets, device)

    train_classifier(cache, device)


if __name__ == "__main__":
    main()

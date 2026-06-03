from pathlib import Path

import random
import numpy as np 
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EyePACSDataset
from src.splits import create_patient_level_split
from src.transfer_learning_model import create_convnext_large_binary_model
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS_CSV = Path("F:/ZPB/dataset/original/trainLabels.csv")
IMAGES_DIR = Path("F:/ZPB/dataset/original/train")

PREPROCESSING = "crop_resize"

IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8
RANDOM_SEED = 42

NUM_WORKERS = 2
PIN_MEMORY = True

OUTPUT_DIR = PROJECT_ROOT / "results" / "transfer_learning" / PREPROCESSING

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df = create_patient_level_split(
        labels_df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_seed=RANDOM_SEED
    )

    train_df = labels_df[labels_df["split"] == "train"].copy()
    val_df = labels_df[labels_df["split"] == "val"].copy()
    test_df = labels_df[labels_df["split"] == "test"].copy()

    train_dataset = EyePACSDataset(
        labels_df=train_df,
        images_dir=IMAGES_DIR,
        preprocessing=PREPROCESSING,
        image_size=IMAGE_SIZE,
        use_augmentations=True
    )

    val_dataset = EyePACSDataset(
        labels_df=val_df,
        images_dir=IMAGES_DIR,
        preprocessing=PREPROCESSING,
        image_size=IMAGE_SIZE,
        use_augmentations=False
    )

    test_dataset = EyePACSDataset(
        labels_df=test_df,
        images_dir=IMAGES_DIR,
        preprocessing=PREPROCESSING,
        image_size=IMAGE_SIZE,
        use_augmentations=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    return train_loader, val_loader, test_loader


def calculate_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5
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
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
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

    return metrics


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress_bar.set_postfix(train_loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, dict[str, float]]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            probabilities = torch.sigmoid(logits)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.append(labels.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())

    labels_np = np.concatenate(all_labels)
    probabilities_np = np.concatenate(all_probabilities)

    metrics = calculate_binary_metrics(labels_np, probabilities_np)
    average_loss = total_loss / total_samples

    return average_loss, metrics


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


def main() -> None:
    set_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    print("Device:", device)
    print("Preprocessing:", PREPROCESSING)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("TensorBoard:")
    print("  tensorboard --logdir results/transfer_learning")

    train_loader, val_loader, test_loader = create_dataloaders()

    model = create_convnext_large_binary_model(pretrained=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(log_dir=OUTPUT_DIR / "tensorboard")

    best_val_roc_auc = -1.0
    epochs_without_improvement = 0

    history = []

    best_model_path = OUTPUT_DIR / "best_model.pt"
    history_path = OUTPUT_DIR / "history.csv"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )

        val_loss, val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )

        print(f"Train loss: {train_loss:.4f}")
        print_metrics("Val", val_loss, val_metrics)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Val/roc_auc", val_metrics["roc_auc"], epoch)
        writer.add_scalar("Val/pr_auc", val_metrics["pr_auc"], epoch)
        writer.add_scalar("Val/balanced_accuracy", val_metrics["balanced_accuracy"], epoch)
        writer.add_scalar("Val/precision", val_metrics["precision"], epoch)
        writer.add_scalar("Val/recall", val_metrics["recall"], epoch)
        writer.add_scalar("Val/specificity", val_metrics["specificity"], epoch)
        writer.add_scalar("Val/f1", val_metrics["f1"], epoch)
        writer.flush()

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()}
        }
        history.append(epoch_result)

        current_val_roc_auc = val_metrics["roc_auc"]

        if current_val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = current_val_roc_auc
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_roc_auc": best_val_roc_auc,
                    "preprocessing": PREPROCESSING,
                    "image_size": IMAGE_SIZE,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                },
                best_model_path
            )

            print(f"Saved new best model: {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE} epochs"
            )

        pd.DataFrame(history).to_csv(history_path, index=False)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nLoading best model for final test evaluation...")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print_metrics("Test", test_loss, test_metrics)

    test_results = {
        "test_loss": test_loss,
        **{f"test_{key}": value for key, value in test_metrics.items()}
    }

    pd.DataFrame([test_results]).to_csv(
        OUTPUT_DIR / "test_results.csv",
        index=False
    )

    print("\nDone.")
    print("Best validation ROC-AUC:", best_val_roc_auc)
    print("Results saved to:", OUTPUT_DIR)

    writer.close()


if __name__ == "__main__":
    main()

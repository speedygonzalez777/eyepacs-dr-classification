"""Five-grade training with binary DR validation and checkpoint selection."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from scripts.train_transfer_learning import (
    EVALUATION_THRESHOLD, FIVE_CLASS_CONFIGS, IMAGES_DIR, MIN_DELTA_VAL_LOSS, RESULTS_DIR, TINY_CONFIG,
    calculate_binary_metrics, create_dataloader, create_optimizer, set_seed, train_one_epoch,
)

GRADES = (0, 1, 2, 3, 4)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SPLIT = PROJECT_ROOT / "metadata" / "convnext_tiny_regularized_split.csv"


def create_grade_model() -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Sequential(
        nn.Dropout(TINY_CONFIG.classifier_dropout),
        nn.Linear(model.classifier[2].in_features, len(GRADES)),
    )
    return model


@torch.no_grad()
def validate_grades(model, loader, device) -> dict[str, float]:
    model.eval()
    ce_sum = bce_sum = 0.0
    grades, predicted_grades, dr_probabilities = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(images)
        logits = logits.float()
        probs = logits.softmax(dim=1)
        # log[P(DR) / P(no DR)] directly from the five logits.
        dr_logits = torch.logsumexp(logits[:, 1:] - logits[:, :1], dim=1)
        ce_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        bce_sum += F.binary_cross_entropy_with_logits(
            dr_logits, (labels > 0).float(), reduction="sum"
        ).item()
        grades.append(labels.cpu().numpy())
        predicted_grades.append(logits.argmax(dim=1).cpu().numpy())
        dr_probabilities.append(probs[:, 1:].sum(dim=1).cpu().numpy())

    targets = np.concatenate(grades)
    predictions = np.concatenate(predicted_grades)
    binary = calculate_binary_metrics(targets > 0, np.concatenate(dr_probabilities))
    return {
        "val_5class_ce": ce_sum / len(targets),
        "val_binary_bce": bce_sum / len(targets),
        "val_5class_accuracy": float(accuracy_score(targets, predictions)),
        "val_5class_macro_f1": float(f1_score(targets, predictions, labels=GRADES,
                                              average="macro", zero_division=0)),
        **{f"val_binary_{key}": float(value) if isinstance(value, np.floating) else value
           for key, value in binary.items()},
    }


def print_epoch_scores(scores: dict[str, float]) -> None:
    print((
        "Train loss: {train_5class_ce:.4f}\n"
        "Val loss: {val_5class_ce:.4f}\n"
        "Binary val loss: {val_binary_bce:.4f}\n"
        "ROC-AUC: {val_binary_roc_auc:.4f} | PR-AUC: {val_binary_pr_auc:.4f} | "
        "Balanced acc: {val_binary_balanced_accuracy:.4f} | Precision: {val_binary_precision:.4f} | "
        "Recall: {val_binary_recall:.4f} | Specificity: {val_binary_specificity:.4f} | F1: {val_binary_f1:.4f}\n"
        "5-class accuracy: {val_5class_accuracy:.4f} | Macro F1: {val_5class_macro_f1:.4f}\n"
        "TN: {val_binary_tn:.4f} | FP: {val_binary_fp:.4f} | FN: {val_binary_fn:.4f} | TP: {val_binary_tp:.4f}"
    ).format_map(scores), flush=True)


def run_five_class(output_dir: Path, experiment: str = "convnext_tiny_5class") -> None:
    config = FIVE_CLASS_CONFIGS[experiment]
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f"{experiment} requires CUDA with native BF16 support.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    set_seed(config.random_seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = output_dir / "split.csv"
    copyfile(REFERENCE_SPLIT, manifest)
    rows = pd.read_csv(manifest, dtype={"patient_id": str})
    if not rows["level"].isin(GRADES).all():
        raise ValueError("The saved split contains an invalid EyePACS grade.")
    train = create_dataloader(rows, "train", config, use_grade_labels=True)
    val = create_dataloader(rows, "val", config, use_grade_labels=True)

    training = asdict(config)
    del training["min_delta_roc_auc"]
    training["backbone_learning_rate"] = training.pop("backbone_last_stage_learning_rate")
    metadata = {
        "experiment": experiment, "run_id": output_dir.name,
        "training": training, "classes": GRADES, "target": "level",
        "pretrained_weights": str(ConvNeXt_Tiny_Weights.DEFAULT),
        "optimizer": "AdamW", "loss": "CrossEntropyLoss", "scheduler": None,
        "amp_dtype": "bfloat16", "cudnn_benchmark": True,
        "checkpoint_monitor": "val_binary_bce", "early_stopping_monitor": "val_binary_bce",
        "min_delta_binary_bce": MIN_DELTA_VAL_LOSS, "binary_threshold": EVALUATION_THRESHOLD,
        "data": {"images_dir": str(IMAGES_DIR), "split_manifest": manifest.name,
                 "source_split": str(REFERENCE_SPLIT), "split_sha256": sha256(manifest.read_bytes()).hexdigest(),
                 "augmentation_splits": ["train"]},
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    model = create_grade_model().to(device)
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=False)
    best_bce = patience_reference = float("inf")
    stale_epochs = 0
    history = []
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Run directory:", output_dir)
    with SummaryWriter(log_dir=output_dir / "tensorboard") as writer:
        for epoch in range(1, config.num_epochs + 1):
            print(f"\nEpoch {epoch}/{config.num_epochs}\nTrainable parameters: {trainable:,}", flush=True)
            model.train()
            train_ce = train_one_epoch(model, train, criterion, optimizer, scaler, device,
                                       amp_dtype=torch.bfloat16)
            metrics = validate_grades(model, val, device)
            scores = {"train_5class_ce": train_ce, **metrics}
            invalid = [name for name, value in scores.items() if not np.isfinite(value)]
            if invalid:
                raise FloatingPointError(f"Non-finite metrics at epoch {epoch}: {', '.join(invalid)}")
            for name, value in scores.items():
                writer.add_scalar(name, value, epoch)
            writer.flush()
            history.append({"epoch": epoch, **scores})
            pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
            print_epoch_scores(scores)

            bce = metrics["val_binary_bce"]
            if bce < best_bce:
                best_bce = bce
                tmp = output_dir / "best_model.pt.tmp"
                torch.save({
                    "experiment": metadata["experiment"], "run_id": output_dir.name,
                    "epoch": epoch, "best_val_binary_bce": bce, "metrics": scores,
                    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                }, tmp)
                tmp.replace(output_dir / "best_model.pt")
                print(f"Saved best checkpoint (binary val loss: {bce:.4f}): {output_dir / 'best_model.pt'}")
            if bce < patience_reference - MIN_DELTA_VAL_LOSS:
                patience_reference = bce
                stale_epochs = 0
            else:
                stale_epochs += 1
                print(f"Epochs without improvement greater than {MIN_DELTA_VAL_LOSS:.4f}: "
                      f"{stale_epochs}/{config.early_stopping_patience}")
                if stale_epochs >= config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}. Best binary val loss: {best_bce:.4f}", flush=True)
                    break

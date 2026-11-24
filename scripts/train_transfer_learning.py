import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import __version__ as torchvision_version
from torchvision.models import ConvNeXt_Base_Weights, ConvNeXt_Tiny_Weights, EfficientNet_B3_Weights
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import EyePACSDataset
from src.numerical_checks import BatchChecks
from src.splits import create_patient_level_split
from src.transfer_learning_model import (
    configure_convnext_classifier_dropout,
    configure_convnext_partial_finetuning,
    create_convnext_base_binary_model,
    create_convnext_large_binary_model,
    create_convnext_tiny_binary_model,
    create_efficientnet_b3_binary_model,
    create_swin_v2_tiny_binary_model,
    efficientnet_b3_finetunable_features,
    set_convnext_trainable_features,
    set_efficientnet_b3_trainable_features,
)
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATASET_ROOT = PROJECT_ROOT / "dataset" / "original"
LABELS_CSV = DATASET_ROOT / "trainLabels.csv"
IMAGES_DIR = DATASET_ROOT / "train"

RESULTS_DIR = PROJECT_ROOT / "results" / "transfer_learning"
RUN_NAME = "convnext_large_crop_clahe_resize_lr5e-5_wd0.01_dropout0.3"
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
EVALUATION_THRESHOLD = 0.5

# (First epoch, first trainable index in torchvision ConvNeXt.features).
# Indices 6 and 4 include the downsampling before stages 4 and 3.
UNFREEZE_SCHEDULE = ((1, 8), (3, 6), (6, 4), (9, 0))
PLATEAU_SETTINGS = {
    "mode": "max",
    "factor": 0.5,
    "patience": 2,
    "threshold": 0.001,
    "threshold_mode": "abs",
    "cooldown": 1,
    "min_lr": 1e-6,
}
BASE_EXPERIMENTS = {
    "base_baseline": {},
    "base_scheduler": {"use_scheduler": True},
    "base_unfreeze": {"training_mode": "gradual_unfreeze"},
    "base_scheduler_unfreeze": {
        "training_mode": "gradual_unfreeze",
        "use_scheduler": True,
    },
}
EFFICIENTNET_EXPERIMENT = "efficientnet_b3_partial"
EFFICIENTNET_FINETUNE_START_EPOCH = 3
TINY_EXPERIMENT = "convnext_tiny_regularized"
TINY_PARTIAL = "convnext_tiny_partial"
TINY_LAST_STAGE = 7
TRAIN_EVAL_SIZE = 1024
MIN_DELTA_VAL_LOSS = 0.001


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "convnext_large"
    training_mode: str = "full_finetune"
    preprocessing: str = "crop_clahe_resize"
    image_size: int = 512
    batch_size: int = 8
    num_epochs: int = 30
    learning_rate: float = 5e-5
    classifier_learning_rate: float = 1e-4
    backbone_last_stage_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    classifier_dropout: float | None = 0.3
    early_stopping_patience: int = 15
    min_delta_roc_auc: float = 0.001
    random_seed: int = 42
    num_workers: int = 3
    pin_memory: bool = True
    dataloader_seed: int | None = None
    use_scheduler: bool = False


DEFAULT_CONFIG = TrainingConfig()
BASE_CONFIG = TrainingConfig(
    model_name="convnext_base",
    batch_size=15,
    learning_rate=1e-4,
    weight_decay=1e-4,
    classifier_dropout=None,
    early_stopping_patience=4,
    min_delta_roc_auc=0.0,
    dataloader_seed=42,
)
EFFICIENTNET_CONFIG = TrainingConfig(
    model_name="efficientnet_b3",
    training_mode=EFFICIENTNET_EXPERIMENT,
    batch_size=15,
    learning_rate=1e-4,
    classifier_learning_rate=1e-4,
    backbone_last_stage_learning_rate=1e-5,
    weight_decay=1e-4,
    classifier_dropout=0.3,
    early_stopping_patience=4,
    min_delta_roc_auc=0.0,
    dataloader_seed=42,
)


EFFICIENTNET_EXPERIMENTS = {
    EFFICIENTNET_EXPERIMENT: EFFICIENTNET_CONFIG,
    "efficientnet_b3_full": replace(EFFICIENTNET_CONFIG, training_mode="full_finetune"),
}
TINY_CONFIG = replace(BASE_CONFIG, model_name="convnext_tiny", weight_decay=0.01,
                      classifier_dropout=0.3)
FIVE_CLASS_CONFIGS = {
    "convnext_tiny_5class": TINY_CONFIG,
    "convnext_tiny_5class_no_clahe": replace(TINY_CONFIG, preprocessing="crop_resize"),
}
NAMED_CONFIGS = {
    **EFFICIENTNET_EXPERIMENTS,
    TINY_EXPERIMENT: TINY_CONFIG,
    TINY_PARTIAL: replace(TINY_CONFIG, training_mode=TINY_PARTIAL),
}


def train_eval_sample(labels_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    train_rows = labels_df.loc[labels_df["split"] == "train"]
    sample, _ = train_test_split(train_rows, train_size=TRAIN_EVAL_SIZE, random_state=seed,
                                 stratify=train_rows["level"])
    return sample.assign(split="train_eval")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloader(
    labels_df: pd.DataFrame,
    split: str,
    config: TrainingConfig,
    images_dir: Path = IMAGES_DIR,
    include_names: bool = False,
    use_grade_labels: bool = False,
) -> DataLoader:
    is_training = split == "train"
    dataset = EyePACSDataset(
        labels_df=labels_df.loc[labels_df["split"] == split],
        images_dir=images_dir,
        preprocessing=config.preprocessing,
        image_size=config.image_size,
        use_augmentations=is_training,
        include_names=include_names,
        use_grade_labels=use_grade_labels,
    )
    # Keep data order and worker seeds independent of stochastic depth in the model.
    generator = None
    if config.dataloader_seed is not None:
        generator = torch.Generator().manual_seed(config.dataloader_seed)

    print(f"{split.capitalize()} samples:", len(dataset))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_training,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=generator,
    )


def create_model(model_name: str, pretrained: bool = True) -> nn.Module:
    model_factories = {
        "convnext_large": create_convnext_large_binary_model,
        "convnext_tiny": create_convnext_tiny_binary_model,
        "convnext_base": create_convnext_base_binary_model,
        "efficientnet_b3": create_efficientnet_b3_binary_model,
        "swin_v2_tiny": create_swin_v2_tiny_binary_model,
    }

    if model_name not in model_factories:
        available_models = ", ".join(model_factories.keys())
        raise ValueError(
            f"Unknown MODEL_NAME: {model_name}. "
            f"Available models: {available_models}"
        )

    return model_factories[model_name](pretrained=pretrained)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


def configure_training_mode(model: nn.Module, config: TrainingConfig) -> nn.Module:
    if config.training_mode == TINY_PARTIAL:
        if config.model_name != "convnext_tiny":
            raise ValueError("convnext_tiny_partial requires ConvNeXt-Tiny.")
        set_convnext_trainable_features(model, TINY_LAST_STAGE)
        return configure_convnext_classifier_dropout(model, dropout=config.classifier_dropout)

    if config.model_name == "efficientnet_b3" and config.training_mode in {
        EFFICIENTNET_EXPERIMENT, "full_finetune",
    }:
        if config.classifier_dropout is not None:
            model.classifier[0].p = config.classifier_dropout
        if config.training_mode == EFFICIENTNET_EXPERIMENT:
            set_efficientnet_b3_trainable_features(model, finetune_backbone=False)
        else:
            model.requires_grad_(True)
        return model

    if config.training_mode == "convnext_partial_finetune":
        if config.model_name != "convnext_large":
            raise ValueError(
                "TRAINING_MODE='convnext_partial_finetune' requires "
                "MODEL_NAME='convnext_large'."
            )
        return configure_convnext_partial_finetuning(model)

    if config.training_mode in {"full_finetune", "gradual_unfreeze"}:
        if config.model_name.startswith("convnext_") and config.classifier_dropout is not None:
            model = configure_convnext_classifier_dropout(
                model,
                dropout=config.classifier_dropout,
            )
        return model

    raise ValueError(
        f"Unknown training mode: {config.training_mode}. "
        "Available modes: full_finetune, convnext_partial_finetune, gradual_unfreeze, "
        f"{EFFICIENTNET_EXPERIMENT}"
    )


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.model_name in {"efficientnet_b3", "convnext_tiny"}:
        backbone_parameters = model.features.parameters()
        if config.training_mode == TINY_PARTIAL:
            backbone_parameters = model.features[TINY_LAST_STAGE].parameters()
        elif config.training_mode == EFFICIENTNET_EXPERIMENT:
            backbone_parameters = [
                parameter
                for block in efficientnet_b3_finetunable_features(model)
                for parameter in block.parameters()
            ]
        return torch.optim.AdamW(
            [
                {"params": model.classifier.parameters(), "lr": config.classifier_learning_rate},
                {"params": backbone_parameters, "lr": config.backbone_last_stage_learning_rate},
            ],
            weight_decay=config.weight_decay,
        )

    if config.training_mode == "convnext_partial_finetune":
        classifier_parameters = [
            parameter
            for parameter in model.classifier.parameters()
            if parameter.requires_grad
        ]
        last_stage_parameters = [
            parameter
            for parameter in model.features[-1].parameters()
            if parameter.requires_grad
        ]

        return torch.optim.AdamW(
            [
                {
                    "params": classifier_parameters,
                    "lr": config.classifier_learning_rate,
                },
                {
                    "params": last_stage_parameters,
                    "lr": config.backbone_last_stage_learning_rate,
                },
            ],
            weight_decay=config.weight_decay,
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def calculate_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = EVALUATION_THRESHOLD,
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
    device: torch.device,
    diagnostic_log: Path | None = None,
    epoch: int | None = None,
    amp_dtype: torch.dtype | None = None,
) -> float:
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch_index, batch in enumerate(progress_bar, 1):
        images, labels = batch[:2]
        checks = None
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if diagnostic_log is not None:
            checks = BatchChecks(diagnostic_log, "train", epoch, batch_index, batch[2])
            if batch_index == 1:
                checks.state("before_forward.parameters", model.named_parameters())
                checks.batch_norm(model, "before_forward")
            checks.tensor("input.images", images, per_image=True)
            checks.tensor("input.labels", labels, per_image=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=amp_dtype):
            logits = model(images).squeeze(1)
            if checks:
                checks.batch_norm(model)
                checks.tensor("forward.logits", logits, per_image=True)
            if isinstance(criterion, nn.CrossEntropyLoss):
                # Keep the CE log-softmax in FP32 even when the head returns BF16.
                logits = logits.float()
            loss = criterion(logits, labels)
            if checks:
                checks.tensor("forward.loss", loss)

        scaler.scale(loss).backward()
        if checks:
            scaler.unscale_(optimizer)
            # GradScaler skips the step and lowers the scale on a backward overflow.
            checks.state("backward.gradients", ((k, p.grad) for k, p in model.named_parameters()),
                         fatal=not scaler.is_enabled(), scale=scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if checks:
            checks.state("optimizer.parameters", model.named_parameters())

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress_bar.set_postfix(train_loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    diagnostic_log: Path | None = None,
    epoch: int | None = None,
    phase: str = "val",
    use_amp: bool = True,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, dict[str, float]]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader, 1):
            images, labels = batch[:2]
            checks = None
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if diagnostic_log is not None:
                checks = BatchChecks(diagnostic_log, phase, epoch, batch_index, batch[2])
                if batch_index == 1:
                    checks.state("before_forward.parameters", model.named_parameters())
                    checks.batch_norm(model, "before_forward")
                checks.tensor("input.images", images, per_image=True)
                checks.tensor("input.labels", labels, per_image=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda", dtype=amp_dtype):
                logits = model(images).squeeze(1)
                if checks:
                    checks.tensor("forward.logits", logits, per_image=True)
                loss = criterion(logits, labels)
                if checks:
                    checks.tensor("forward.loss", loss)

            # Avoid extra ranking ties when AMP produces half-precision logits.
            probabilities = torch.sigmoid(logits.float())

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


def save_experiment_config(
    output_dir: Path,
    experiment: str,
    config: TrainingConfig,
    device: torch.device,
    early_stopping_start_epoch: int,
) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    source_files = (
        "scripts/train_transfer_learning.py",
        "src/dataset.py",
        "src/preprocessing.py",
        "src/splits.py",
        "src/transfer_learning_model.py",
        "requirements.txt",
    )
    if experiment == "efficientnet_b3_full":
        source_files += ("src/numerical_checks.py",)
    efficientnet_partial = config.training_mode == EFFICIENTNET_EXPERIMENT
    pretrained_weights = (
        EfficientNet_B3_Weights.DEFAULT if config.model_name == "efficientnet_b3" else ConvNeXt_Base_Weights.DEFAULT
    )
    metadata = {
        "run_id": output_dir.name,
        "experiment": experiment,
        "training": asdict(config),
        "pretrained_weights": str(pretrained_weights),
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "scheduler": (
            {"name": "ReduceLROnPlateau", "monitor": "val_roc_auc", **PLATEAU_SETTINGS}
            if config.use_scheduler else None
        ),
        "unfreeze_schedule": [
            {"first_epoch": epoch, "first_trainable_feature": index}
            for epoch, index in UNFREEZE_SCHEDULE
        ] if config.training_mode == "gradual_unfreeze" else None,
        "early_stopping_start_epoch": early_stopping_start_epoch,
        "data": {
            "labels_csv": str(LABELS_CSV),
            "images_dir": str(IMAGES_DIR),
            "split_manifest": "split.csv",
            "split_fractions": dict(zip(("train", "val", "test"), SPLIT_FRACTIONS)),
            "augmentation_splits": ["train"],
        },
        "evaluation": {
            "checkpoint_monitor": "val_roc_auc",
            "threshold": EVALUATION_THRESHOLD,
            "probability_dtype": "float32",
        },
        "environment": {
            "device": str(device),
            "amp": device.type == "cuda",
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision_version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "git_commit": commit,
        "source_sha256": {
            name: sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in source_files
        },
    }
    if efficientnet_partial:
        metadata["unfreeze_schedule"] = [
            {"first_epoch": 1, "trainable_modules": ["classifier"]},
            {
                "first_epoch": EFFICIENTNET_FINETUNE_START_EPOCH,
                "trainable_modules": ["classifier", "features[-2][-1]", "features[-1]"],
            },
        ]
    if config.model_name == "efficientnet_b3":
        metadata["evaluation"]["early_stopping_monitor"] = "val_roc_auc"
        metadata["environment"]["amp_dtype"] = "float16" if device.type == "cuda" else None
    if experiment == "efficientnet_b3_full":
        metadata["evaluation"]["forward_dtype"] = "float32"
    if experiment in {TINY_EXPERIMENT, TINY_PARTIAL}:
        metadata["pretrained_weights"] = str(ConvNeXt_Tiny_Weights.DEFAULT)
        metadata["environment"]["amp_dtype"] = "bfloat16"
        metadata["evaluation"].update(checkpoint_monitor="val_loss", early_stopping_monitor="val_loss",
                                      min_delta_val_loss=MIN_DELTA_VAL_LOSS)
        metadata["data"].update(train_eval_manifest="train_eval.csv", train_eval_size=TRAIN_EVAL_SIZE,
                                train_eval_stratify="level")
    if experiment == TINY_PARTIAL:
        metadata["trainable_modules"] = [f"features[{TINY_LAST_STAGE}]", "classifier"]
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def run_training(
    config: TrainingConfig,
    output_dir: Path,
    experiment: str | None = None,
) -> None:
    set_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tiny = config.model_name == "convnext_tiny"
    tiny_partial = config.training_mode == TINY_PARTIAL
    if tiny and (device.type != "cuda" or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise RuntimeError(f"{experiment or config.model_name} requires CUDA with native BF16 support.")
    amp_dtype = torch.bfloat16 if tiny else None
    output_dir.mkdir(parents=True, exist_ok=False)
    efficientnet_partial = config.training_mode == EFFICIENTNET_EXPERIMENT
    torch.backends.cudnn.benchmark = efficientnet_partial or device.type == "cuda"
    gradual_unfreeze = config.training_mode == "gradual_unfreeze"
    # Select checkpoints from every stage, but count patience only during full fine-tuning.
    early_stopping_start_epoch = UNFREEZE_SCHEDULE[-1][0] if gradual_unfreeze else 1
    if efficientnet_partial:
        early_stopping_start_epoch = EFFICIENTNET_FINETUNE_START_EPOCH

    print("Device:", device)
    print("Model:", config.model_name)
    print("Training mode:", config.training_mode)
    print("Preprocessing:", config.preprocessing)
    print("Image size:", config.image_size)
    print("Batch size:", config.batch_size)
    print("Learning rate:", config.learning_rate)
    print("Weight decay:", config.weight_decay)
    print("Classifier dropout:", config.classifier_dropout)
    print("Epochs:", config.num_epochs)
    print("Output directory:", output_dir)
    print("TensorBoard:")
    print("  tensorboard --logdir results/transfer_learning")

    train_size, val_size, test_size = SPLIT_FRACTIONS
    labels_df = create_patient_level_split(
        pd.read_csv(LABELS_CSV),
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_seed=config.random_seed,
    )
    if experiment is not None:
        save_experiment_config(output_dir, experiment, config, device, early_stopping_start_epoch)
        labels_df[["image", "patient_id", "level", "binary_label", "split"]].to_csv(
            output_dir / "split.csv", index=False
        )

    full_b3 = config.model_name == "efficientnet_b3" and config.training_mode == "full_finetune"
    diagnostic_log = output_dir / "numerics.jsonl" if full_b3 else None
    train_loader = create_dataloader(labels_df, "train", config, include_names=full_b3)
    val_loader = create_dataloader(labels_df, "val", config, include_names=full_b3)
    train_eval_loader = None
    if tiny:
        sample = train_eval_sample(labels_df, config.random_seed)
        sample[["image", "patient_id", "level", "binary_label"]].to_csv(output_dir / "train_eval.csv", index=False)
        train_eval_loader = create_dataloader(sample, "train_eval", config)

    model = create_model(config.model_name, pretrained=True)
    model = configure_training_mode(model, config)
    model = model.to(device)
    total_parameters, _ = count_parameters(model)
    print("Total parameters:", total_parameters)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = create_optimizer(model, config)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **PLATEAU_SETTINGS)
        if config.use_scheduler else None
    )

    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and not tiny)
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")

    best_val_roc_auc = -1.0
    best_val_loss = float("inf")
    patience_reference_loss = float("inf")
    epochs_without_improvement = 0

    history = []

    best_model_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.csv"

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")

        model.train()
        first_trainable_feature = 0
        early_stopping_active = epoch >= early_stopping_start_epoch
        if gradual_unfreeze:
            first_trainable_feature = next(
                index for start, index in reversed(UNFREEZE_SCHEDULE) if epoch >= start
            )
            set_convnext_trainable_features(model, first_trainable_feature)
        elif tiny_partial:
            # model.train() also enables stochastic depth in frozen stages.
            first_trainable_feature = TINY_LAST_STAGE
            set_convnext_trainable_features(model, first_trainable_feature)
        elif efficientnet_partial:
            set_efficientnet_b3_trainable_features(
                model, finetune_backbone=epoch >= EFFICIENTNET_FINETUNE_START_EPOCH
            )
        _, trainable_parameters = count_parameters(model)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"Trainable parameters: {trainable_parameters:,}/{total_parameters:,}")

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device, diagnostic_log=diagnostic_log, epoch=epoch,
            amp_dtype=amp_dtype,
        )

        val_loss, val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device, diagnostic_log=diagnostic_log, epoch=epoch,
            use_amp=not full_b3,
            amp_dtype=amp_dtype,
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
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()}
        }
        if experiment is not None:
            epoch_result.update(
                learning_rate=learning_rate,
                trainable_parameters=trainable_parameters,
            )
            if config.model_name == "efficientnet_b3" or tiny:
                backbone_lr = optimizer.param_groups[1]["lr"] if early_stopping_active else 0.0
                epoch_result["backbone_learning_rate"] = backbone_lr
                writer.add_scalar("Train/backbone_learning_rate", backbone_lr, epoch)
            else:
                epoch_result["first_trainable_feature"] = first_trainable_feature
            epoch_result["early_stopping_active"] = early_stopping_active
            writer.add_scalar("Train/learning_rate", learning_rate, epoch)
            writer.add_scalar("Train/trainable_parameters", trainable_parameters, epoch)
        if train_eval_loader is not None:
            train_eval_loss, _ = evaluate(model, train_eval_loader, criterion, device, amp_dtype=amp_dtype)
            print(f"Train eval loss: {train_eval_loss:.4f}")
            epoch_result["train_eval_loss"] = train_eval_loss
            writer.add_scalar("Loss/train_eval", train_eval_loss, epoch)
            if tiny_partial:
                loss_gap = val_loss - train_eval_loss
                epoch_result["val_minus_train_eval_loss"] = loss_gap
                writer.add_scalar("Loss/val_minus_train_eval", loss_gap, epoch)
                print(f"Val loss minus train eval loss: {loss_gap:+.4f}")
        writer.flush()
        history.append(epoch_result)

        current_val_roc_auc = float(val_metrics["roc_auc"])
        if scheduler is not None:
            scheduler.step(current_val_roc_auc)
            next_learning_rate = optimizer.param_groups[0]["lr"]
            if next_learning_rate != learning_rate:
                print(f"Learning rate for next epoch: {next_learning_rate:g}")

        if tiny:
            save_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            reset_patience = val_loss < patience_reference_loss - MIN_DELTA_VAL_LOSS
            if reset_patience:
                patience_reference_loss = val_loss
        else:
            save_best = current_val_roc_auc > best_val_roc_auc + config.min_delta_roc_auc
            reset_patience = save_best
            if save_best:
                best_val_roc_auc = current_val_roc_auc
        if reset_patience:
            epochs_without_improvement = 0
        if save_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_name": config.model_name,
                "training_mode": config.training_mode,
                "preprocessing": config.preprocessing,
                "image_size": config.image_size,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "classifier_learning_rate": config.classifier_learning_rate,
                "backbone_last_stage_learning_rate": config.backbone_last_stage_learning_rate,
                "weight_decay": config.weight_decay,
                "min_delta_roc_auc": config.min_delta_roc_auc,
            }
            if tiny:
                checkpoint.update(best_val_loss=val_loss, val_roc_auc=current_val_roc_auc,
                                  min_delta_val_loss=MIN_DELTA_VAL_LOSS)
            else:
                checkpoint["best_val_roc_auc"] = best_val_roc_auc
            if experiment is not None:
                checkpoint.update(
                    experiment=experiment,
                    run_id=output_dir.name,
                    scaler_state_dict=scaler.state_dict(),
                )
                if not efficientnet_partial:
                    checkpoint["first_trainable_feature"] = first_trainable_feature
                if scheduler is not None:
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            # Replace the previous best only once the new checkpoint is fully written.
            tmp_model_path = best_model_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, tmp_model_path)
            tmp_model_path.replace(best_model_path)

            print(f"Saved new best model: {best_model_path}")
        if not reset_patience and early_stopping_active:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/"
                f"{config.early_stopping_patience} epochs"
            )

        pd.DataFrame(history).to_csv(history_path, index=False)

        if epochs_without_improvement >= config.early_stopping_patience:
            print("Early stopping triggered.")
            break

    writer.close()
    if not best_model_path.is_file():
        raise RuntimeError("Training produced no checkpoint with a valid validation score.")

    if experiment is None:
        test_loader = create_dataloader(labels_df, "test", config)
        evaluate_checkpoint(model, test_loader, device, output_dir)

    print("\nDone.")
    if tiny:
        print("Best validation loss:", best_val_loss)
    else:
        print("Best validation ROC-AUC:", best_val_roc_auc)
    print("Results saved to:", output_dir)


def evaluate_checkpoint(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    run_id: str | None = None,
) -> None:
    print("\nLoading best model for final test evaluation...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    if run_id is not None and checkpoint.get("run_id") != run_id:
        raise ValueError("The checkpoint's run ID does not match the configuration.")
    model.load_state_dict(checkpoint["model_state_dict"])

    full_b3 = (checkpoint.get("model_name"), checkpoint.get("training_mode")) == (
        "efficientnet_b3", "full_finetune"
    )
    test_loss, test_metrics = evaluate(
        model, test_loader, nn.BCEWithLogitsLoss(), device, use_amp=not full_b3,
        amp_dtype=torch.bfloat16 if checkpoint.get("model_name") == "convnext_tiny" else None,
    )

    print_metrics("Test", test_loss, test_metrics)

    test_results = {
        "test_loss": test_loss,
        **{f"test_{key}": value for key, value in test_metrics.items()}
    }
    if run_id is not None:
        test_results.update(run_id=run_id, checkpoint_epoch=checkpoint["epoch"])

    pd.DataFrame([test_results]).to_csv(
        output_dir / "test_results.csv", index=False, mode="x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EyePACS transfer-learning models.")
    mode = parser.add_mutually_exclusive_group()
    experiments = (*BASE_EXPERIMENTS, *NAMED_CONFIGS)
    mode.add_argument("--experiment", choices=(*experiments, *FIVE_CLASS_CONFIGS), help="Train a named experiment.")
    mode.add_argument("--evaluate-test", type=Path, metavar="RUN_DIR", help="Test a selected experiment run.")
    args = parser.parse_args()

    if args.evaluate_test is not None:
        output_dir = args.evaluate_test
        if (output_dir / "test_results.csv").exists():
            parser.error(f"Test results already exist in {output_dir}.")
        metadata = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        if metadata["experiment"] not in experiments:
            parser.error("--evaluate-test requires a named experiment directory.")
        config = TrainingConfig(**metadata["training"])
        labels_df = pd.read_csv(output_dir / metadata["data"]["split_manifest"])
        test_loader = create_dataloader(labels_df, "test", config, Path(metadata["data"]["images_dir"]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = device.type == "cuda"
        model = configure_training_mode(create_model(config.model_name, pretrained=False), config)
        evaluate_checkpoint(model.to(device), test_loader, device, output_dir, metadata["run_id"])
    elif args.experiment is not None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{uuid4().hex[:8]}"
        if args.experiment in FIVE_CLASS_CONFIGS:
            from scripts.train_convnext_tiny_5class import run_five_class

            run_five_class(RESULTS_DIR / args.experiment / run_id, args.experiment)
            return
        config = (
            NAMED_CONFIGS[args.experiment] if args.experiment in NAMED_CONFIGS
            else replace(BASE_CONFIG, **BASE_EXPERIMENTS[args.experiment])
        )
        run_training(config, RESULTS_DIR / args.experiment / run_id, args.experiment)
    else:
        run_training(DEFAULT_CONFIG, RESULTS_DIR / RUN_NAME)


if __name__ == "__main__":
    main()

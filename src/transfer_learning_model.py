import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_B3_Weights,
    Swin_V2_T_Weights,
    convnext_base,
    convnext_large,
    convnext_tiny,
    efficientnet_b3,
    swin_v2_t,
)


class _CheckpointedStage(nn.Sequential):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            return checkpoint(super().forward, x, use_reentrant=False)
        return super().forward(x)


def create_convnext_large_binary_model(pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Large_Weights.DEFAULT if pretrained else None

    model = convnext_large(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)

    return model


def create_convnext_tiny_binary_model(pretrained: bool = True) -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    return model


def create_convnext_base_binary_model(pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None

    model = convnext_base(weights=weights)
    # Keep the original block indices and state_dict keys under features.5.
    model.features[5] = _CheckpointedStage(*model.features[5])

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)

    return model


def create_swin_v2_tiny_binary_model(pretrained: bool = True) -> nn.Module:
    weights = Swin_V2_T_Weights.DEFAULT if pretrained else None

    model = swin_v2_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, 1)

    return model


def create_efficientnet_b3_binary_model(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = efficientnet_b3(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def efficientnet_b3_finetunable_features(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    # The penultimate stage contains multiple MBConv blocks. Select only its last block.
    return model.features[-2][-1], model.features[-1]


def set_efficientnet_b3_trainable_features(model: nn.Module, finetune_backbone: bool) -> None:
    model.features.requires_grad_(False)
    model.features.eval()
    for block in efficientnet_b3_finetunable_features(model):
        block.requires_grad_(finetune_backbone)
        block.train(finetune_backbone and model.training)


def configure_convnext_classifier_dropout(
    model: nn.Module,
    dropout: float = 0.3,
) -> nn.Module:
    if not hasattr(model, "classifier") or len(model.classifier) < 3:
        raise ValueError("Expected a torchvision ConvNeXt model with `classifier`.")

    current_head = model.classifier[2]
    if not isinstance(current_head, nn.Linear):
        raise ValueError("Expected ConvNeXt classifier[2] to be nn.Linear.")

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(current_head.in_features, 1),
    )
    return model


def set_convnext_trainable_features(model: nn.Module, first_trainable: int) -> None:
    for index, block in enumerate(model.features):
        trainable = index >= first_trainable
        block.requires_grad_(trainable)
        block.train(trainable)


def configure_convnext_partial_finetuning(model: nn.Module) -> nn.Module:
    if not hasattr(model, "features") or len(model.features) == 0:
        raise ValueError("Expected a torchvision ConvNeXt model with `features`.")
    if not hasattr(model, "classifier") or len(model.classifier) < 3:
        raise ValueError("Expected a torchvision ConvNeXt model with `classifier`.")

    for parameter in model.parameters():
        parameter.requires_grad = False

    last_stage = model.features[-1]
    for parameter in last_stage.parameters():
        parameter.requires_grad = True

    model = configure_convnext_classifier_dropout(model, dropout=0.3)
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Partial fine-tuning enabled.")
    print("Trainable parts: classifier, features[-1] (last ConvNeXt stage)")
    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    return model

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"


class DINOv3LinearProbe(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        super().__init__()

        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("DINOv3 model config does not expose `hidden_size`.")

        self.hidden_size = int(hidden_size)
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True) -> "DINOv3LinearProbe":
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)

        if outputs.pooler_output is None:
            raise ValueError("DINOv3 output does not contain `pooler_output`.")

        return outputs.pooler_output

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.extract_embeddings(pixel_values)
        return self.classifier(embeddings)

    def parameter_summary(self) -> dict[str, int | bool]:
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        classifier_parameters = sum(
            parameter.numel()
            for parameter in self.classifier.parameters()
        )
        backbone_frozen = all(
            not parameter.requires_grad
            for parameter in self.backbone.parameters()
        )

        return {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "classifier_parameters": classifier_parameters,
            "backbone_frozen": backbone_frozen,
        }


def print_parameter_summary(model: DINOv3LinearProbe) -> None:
    summary = model.parameter_summary()
    print("Total parameters:", summary["total_parameters"])
    print("Trainable parameters:", summary["trainable_parameters"])
    print("Classifier parameters:", summary["classifier_parameters"])
    print("Backbone frozen:", summary["backbone_frozen"])

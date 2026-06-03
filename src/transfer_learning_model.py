from torch import nn 
from torchvision.models import ConvNeXt_Large_Weights, convnext_large


def create_convnext_large_binary_model(pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Large_Weights.DEFAULT if pretrained else None 

    model = convnext_large(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)

    return model 
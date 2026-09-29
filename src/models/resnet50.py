"""
ResNet50 Model Architecture for Road Surface Condition Monitoring.

This module provides a PyTorch implementation of the ResNet50 model
pretrained on ImageNet, adapted for multi-class road surface condition classification.

Classes:
    - D00: Longitudinal Crack
    - D10: Transverse Crack
    - D20: Alligator Crack
    - D40: Pothole
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50(
    num_classes: int = 4,
    pretrained: bool = True
) -> nn.Module:
    """
    Create and return a ResNet50 model configured for road surface condition classification.

    The original fully connected classification layer (model.fc) is replaced with
    a new linear layer matching the target number of output classes.

    Args:
        num_classes (int): Number of target output classes. Default is 4.
        pretrained (bool): If True, uses ImageNet pretrained weights (ResNet50_Weights.DEFAULT).
                           If False, initializes weights randomly (weights=None). Default is True.

    Returns:
        nn.Module: PyTorch ResNet50 model adapted for `num_classes` outputs.
    """
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_resnet50_model(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.2
) -> nn.Module:
    """
    Instantiate and return a ResNet50 model for road surface condition monitoring.

    Args:
        num_classes (int): Number of target output classes. Default 4.
        pretrained (bool): Whether to load ImageNet pretrained weights. Default True.
        freeze_backbone (bool): If True, freezes feature extractor parameters. Default False.
        dropout_rate (float): Dropout probability added before FC layer. Default 0.2.

    Returns:
        nn.Module: Configured ResNet50 model.
    """
    model = create_resnet50(num_classes=num_classes, pretrained=pretrained)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    if dropout_rate > 0.0:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    return model

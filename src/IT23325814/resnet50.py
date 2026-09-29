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

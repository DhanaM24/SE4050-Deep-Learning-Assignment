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

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional


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


class ResNet50RoadClassifier(nn.Module):
    """
    ResNet50 classification model wrapper for road surface condition monitoring.

    Args:
        num_classes (int): Number of target output classes. Default is 4.
        pretrained (bool): Whether to use ImageNet pretrained weights. Default is True.
        freeze_backbone (bool): If True, freezes backbone parameters. Default is False.
        dropout_rate (float): Dropout probability before final FC layer. Default is 0.2.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2
    ):
        super(ResNet50RoadClassifier, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features

        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ResNet50RoadClassifier.

        Args:
            x (torch.Tensor): Input batch tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Class logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)


def get_resnet50_model(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.2
) -> ResNet50RoadClassifier:
    """
    Instantiate and return a ResNet50 model for road surface condition monitoring.

    Args:
        num_classes (int): Number of target output classes. Default 4.
        pretrained (bool): Load ImageNet pretrained weights. Default True.
        freeze_backbone (bool): Freeze feature extractor parameters. Default False.
        dropout_rate (float): Dropout probability before FC layer. Default 0.2.

    Returns:
        ResNet50RoadClassifier: Initialized model instance.
    """
    return ResNet50RoadClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate
    )

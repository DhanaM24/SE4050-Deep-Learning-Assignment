"""
Utility functions for RDD2020 Road Surface Condition Monitoring.
Contains dataset transforms, random seed setup, device selection, and helper functions.
"""

import os
import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from typing import Tuple, Dict, Any, Optional

# ImageNet normalization standard statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default RDD2020 4-Class Map
RDD2020_CLASSES = {
    0: "D00 (Longitudinal Crack)",
    1: "D10 (Transverse Crack)",
    2: "D20 (Alligator Crack)",
    3: "D40 (Pothole)"
}


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed number to set. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get computation device (CUDA GPU if available, else CPU).

    Returns:
        torch.device: Torch device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_data_transforms(img_size: Tuple[int, int] = (224, 224)) -> Dict[str, transforms.Compose]:
    """
    Create PyTorch data transformations for training and validation/testing.

    Args:
        img_size (Tuple[int, int]): Target image resolution (height, width). Default (224, 224).

    Returns:
        Dict[str, transforms.Compose]: Dictionary with 'train', 'val', and 'test' transforms.
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return {
        "train": train_transform,
        "val": val_transform,
        "test": val_transform
    }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (224, 224)
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test splits.
    Dataset path is configurable via `data_dir`.

    Expected directory structure:
      data_dir/
        train/
          D00/
          D10/ ...
        val/
        test/

    Args:
        data_dir (str): Root path to the dataset directory.
        batch_size (int): Batch size for training and evaluation. Default 32.
        num_workers (int): Number of sub-processes for data loading. Default 4.
        img_size (Tuple[int, int]): Image target dimensions. Default (224, 224).

    Returns:
        Dict[str, DataLoader]: DataLoaders dictionary for available splits.
    """
    data_transforms = get_data_transforms(img_size)
    dataloaders = {}

    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            is_train = (split == "train")
            dataset = ImageFolder(root=split_path, transform=data_transforms[split])
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=is_train,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            print(f"Loaded '{split}' split from '{split_path}' with {len(dataset)} images across {len(dataset.classes)} classes.")

    return dataloaders

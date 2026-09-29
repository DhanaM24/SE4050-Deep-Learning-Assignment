"""
Evaluation and Metrics Utilities for Road Surface Damage Classification.
Provides model evaluation loops, confusion matrix plotting, and classification report generation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, List, Tuple, Optional


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate a PyTorch classification model on a given dataset loader.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): Data loader for evaluation split.
        criterion (nn.Module): Loss function criterion.
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]:
            - avg_loss: Average loss over the dataset.
            - accuracy: Classification accuracy.
            - all_targets: Array of true target labels.
            - all_preds: Array of predicted target labels.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total_samples = len(dataloader.dataset)
    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0

    return avg_loss, accuracy, all_targets, all_preds


def generate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> pd.DataFrame:
    """
    Generate a detailed metrics summary table (Precision, Recall, F1-Score).

    Args:
        y_true (np.ndarray): Array of ground truth labels.
        y_pred (np.ndarray): Array of predicted labels.
        class_names (List[str]): List of target class names.

    Returns:
        pd.DataFrame: DataFrame containing per-class metrics and summary averages.
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    df_report = pd.DataFrame(report_dict).transpose()
    return df_report


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "ResNet50 Confusion Matrix"
) -> plt.Figure:
    """
    Plot and optionally save a visual confusion matrix heatmap.

    Args:
        y_true (np.ndarray): True ground truth labels.
        y_pred (np.ndarray): Model prediction labels.
        class_names (List[str]): Target class names.
        save_path (str, optional): Destination file path for saving figure.
        title (str): Plot title.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix figure to: {save_path}")

    return fig

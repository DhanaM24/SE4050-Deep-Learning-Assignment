"""
Training Pipeline for ResNet50 Road Surface Damage Classifier.

This script manages end-to-end model training, validation, early stopping,
checkpoint saving, training history logging, learning curve generation,
and post-training validation metrics generation on Train and Val splits.

Usage:
    python src/train.py --data-dir data/classification --epochs 20 --batch-size 32
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet50 import get_resnet50_model
from src.utils import set_seed, get_device, create_dataloaders
from src.evaluate import (
    evaluate_model,
    generate_classification_metrics,
    plot_and_save_confusion_matrix,
)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Train model for one epoch over the dataset loader.

    Args:
        model: PyTorch classification model.
        dataloader: DataLoader for training split.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Computation device (CPU or CUDA).
        max_batches: Optional maximum number of batches to process (for debugging/smoke testing).

    Returns:
        Tuple[float, float]: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%" if total > 0 else "0%",
            }
        )

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_split(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a given data loader with optional max_batches cap.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: (avg_loss, accuracy, targets, preds)
    """
    if max_batches is None:
        return evaluate_model(model, dataloader, criterion, device)

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
            if batch_idx > max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0

    return avg_loss, accuracy, all_targets, all_preds


def plot_learning_curves(
    history_df: pd.DataFrame,
    figures_dir: Path,
) -> Tuple[Path, Path]:
    """
    Plot and save training and validation loss & accuracy curves.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Loss Curve
    fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
    ax_loss.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", marker="o", linewidth=2)
    ax_loss.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", marker="s", linewidth=2)
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.set_title("ResNet50 Training & Validation Loss", fontsize=14, fontweight="bold")
    ax_loss.legend(fontsize=11)
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    loss_curve_path = figures_dir / "loss_curve.png"
    fig_loss.savefig(loss_curve_path, dpi=300)
    plt.close(fig_loss)

    # Accuracy Curve
    fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
    ax_acc.plot(history_df["epoch"], history_df["train_acc"], label="Train Accuracy", marker="o", linewidth=2)
    ax_acc.plot(history_df["epoch"], history_df["val_acc"], label="Val Accuracy", marker="s", linewidth=2)
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Accuracy", fontsize=12)
    ax_acc.set_title("ResNet50 Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax_acc.legend(fontsize=11)
    ax_acc.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    acc_curve_path = figures_dir / "accuracy_curve.png"
    fig_acc.savefig(acc_curve_path, dpi=300)
    plt.close(fig_acc)

    return loss_curve_path, acc_curve_path


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 Road Damage Classifier")
    parser.add_argument("--data-dir", type=str, default="data/classification", help="Path to classification dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Base directory for output artifacts")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Cap train batches per epoch (for testing)")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Cap val batches per epoch (for testing)")

    args = parser.parse_args()

    # 1. Reproducibility & Device
    set_seed(args.seed)
    device = get_device()

    data_dir_path = Path(args.data_dir)
    train_dir = data_dir_path / "train"
    val_dir = data_dir_path / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset directories missing. Checked '{train_dir}' and '{val_dir}'. "
            "Please ensure preprocessing output is present."
        )

    # 2. DataLoaders
    print(f"\nLoading dataset from: {data_dir_path}")
    dataloaders = create_dataloaders(
        data_dir=str(data_dir_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(224, 224),
    )

    if "train" not in dataloaders or "val" not in dataloaders:
        raise ValueError("DataLoaders dictionary must contain 'train' and 'val' splits.")

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    class_names = train_loader.dataset.classes
    num_classes = len(class_names)
    print(f"Dataset classes ({num_classes}): {class_names}")
    print(f"Class mapping (class_to_idx): {train_loader.dataset.class_to_idx}")

    # Verify class ordering matches intended D00, D10, D20, D40 mapping
    expected_classes = ["D00", "D10", "D20", "D40"]
    assert class_names == expected_classes, f"Expected {expected_classes}, got {class_names}"

    # 3. Model Setup
    model = get_resnet50_model(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False,
        dropout_rate=0.2,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"ResNet50 Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    # 4. Criterion, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # Output directory paths
    models_dir = Path(args.output_dir) / "models" / "resnet50"
    tables_dir = Path(args.output_dir) / "tables" / "resnet50"
    figures_dir = Path(args.output_dir) / "figures" / "resnet50"

    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / "best_model.pth"

    # Training state tracking
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history = []
    start_time_total = time.time()

    print("\nStarting Training...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
        )

        # Validate on validation set ONLY
        val_loss, val_acc, _, _ = evaluate_split(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )

        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Step Learning Rate Scheduler
        scheduler.step(val_loss)

        # Log history
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "epoch_time": epoch_duration,
            }
        )

        # Epoch Summary (Concise output as requested)
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")
        print(f"LR: {current_lr:.6f} | Time: {epoch_duration:.2f}s")

        # Save Best Model (based ONLY on validation performance)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "seed": args.seed,
                "training_config": vars(args),
            }
            torch.save(checkpoint, best_model_path)
            print(f"--> Saved best model checkpoint to {best_model_path}")
        else:
            patience_counter += 1
            print(f"--> No validation loss improvement ({patience_counter}/{args.patience})")

        print("-" * 60)

        # Early Stopping Check
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    total_training_duration = time.time() - start_time_total

    # Save History CSV
    history_df = pd.DataFrame(history)
    history_csv_path = tables_dir / "training_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"\nSaved training history to {history_csv_path}")

    # Plot Learning Curves
    loss_curve_path, acc_curve_path = plot_learning_curves(history_df, figures_dir)
    print(f"Saved loss curve to {loss_curve_path}")
    print(f"Saved accuracy curve to {acc_curve_path}")

    # ==================================================
    # STEP 6 — FINAL RESNET50 VALIDATION ON TRAIN & VAL
    # ==================================================
    print("\nRunning post-training evaluation on TRAIN and VAL splits using best checkpoint...")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 1. Evaluate on TRAIN
    train_eval_loss, train_eval_acc, train_targets, train_preds = evaluate_split(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        max_batches=args.max_train_batches,
    )
    df_train_metrics = generate_classification_metrics(train_targets, train_preds, class_names)
    train_metrics_path = tables_dir / "train_metrics.csv"
    df_train_metrics.to_csv(train_metrics_path)
    print(f"Saved TRAIN metrics to {train_metrics_path}")

    train_cm_path = figures_dir / "train_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        y_true=train_targets,
        y_pred=train_preds,
        class_names=class_names,
        save_path=str(train_cm_path),
        title="ResNet50 Train Confusion Matrix",
    )
    plt.close("all")

    # 2. Evaluate on VAL
    val_eval_loss, val_eval_acc, val_targets, val_preds = evaluate_split(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        max_batches=args.max_val_batches,
    )
    df_val_metrics = generate_classification_metrics(val_targets, val_preds, class_names)
    val_metrics_path = tables_dir / "val_metrics.csv"
    df_val_metrics.to_csv(val_metrics_path)
    print(f"Saved VAL metrics to {val_metrics_path}")

    val_cm_path = figures_dir / "val_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        y_true=val_targets,
        y_pred=val_preds,
        class_names=class_names,
        save_path=str(val_cm_path),
        title="ResNet50 Validation Confusion Matrix",
    )
    plt.close("all")

    # Experiment Summary JSON
    summary_data = {
        "model_name": "ResNet50",
        "pretrained": True,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_trained": len(history),
        "seed": args.seed,
        "best_val_accuracy": float(val_eval_acc),
        "best_val_loss": float(val_eval_loss),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "training_duration_sec": total_training_duration,
        "class_mapping": train_loader.dataset.class_to_idx,
    }
    summary_json_path = tables_dir / "experiment_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved experiment summary JSON to {summary_json_path}")

    print("\n" + "=" * 60)
    print("RESNET50 TRAINING & VALIDATION COMPLETE")
    print(f"Best Validation Accuracy: {val_eval_acc * 100:.2f}%")
    print(f"Best Validation Loss:     {val_eval_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

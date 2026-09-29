"""
Final Held-Out Test Set Evaluation for ResNet50 Classifier.

This script evaluates the trained ResNet50 model (best validation checkpoint)
on the held-out test dataset (data/classification/test) ONLY ONCE.

Output files saved:
  - results/tables/resnet50/test_metrics.csv
  - results/tables/resnet50/test_classification_report.csv
  - results/tables/resnet50/test_summary.json
  - results/figures/resnet50/test_confusion_matrix.png
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet50 import get_resnet50_model
from src.utils import set_seed, get_device, create_dataloaders
from src.evaluate import evaluate_model, plot_and_save_confusion_matrix


def run_test_evaluation(
    data_dir: str = "data/classification",
    checkpoint_path: str = "results/models/resnet50/best_model.pth",
    output_dir: str = "results",
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Run final held-out test evaluation on the test set split.
    """
    set_seed(seed)
    device = get_device()

    data_path = Path(data_dir)
    test_dir = data_path / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dataset directory missing at: {test_dir}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best model checkpoint missing at: {ckpt_path}")

    # Load DataLoaders
    dataloaders = create_dataloaders(
        data_dir=str(data_path),
        batch_size=32,
        num_workers=num_workers,
        img_size=(224, 224),
    )

    if "test" not in dataloaders:
        raise ValueError("DataLoader for 'test' split not found.")

    test_loader = dataloaders["test"]
    dataset = test_loader.dataset
    class_names = dataset.classes
    class_to_idx = dataset.class_to_idx

    print(f"Loaded Test Dataset from: {test_dir}")
    print(f"Test dataset size: {len(dataset)} samples")
    print(f"Class names: {class_names}")
    print(f"Class mapping: {class_to_idx}")

    # Expected class verification
    expected_classes = ["D00", "D10", "D20", "D40"]
    assert class_names == expected_classes, f"Class names mismatch! Expected {expected_classes}, got {class_names}"

    # Expected sample counts verification
    expected_counts = {"D00": 946, "D10": 697, "D20": 1238, "D40": 788}
    actual_counts = {}
    for idx, c_name in enumerate(class_names):
        count = sum(1 for _, label in dataset.samples if label == idx)
        actual_counts[c_name] = count

    print(f"Actual per-class sample counts: {actual_counts}")
    print(f"Expected per-class sample counts: {expected_counts}")
    for c_name, exp_cnt in expected_counts.items():
        assert actual_counts[c_name] == exp_cnt, f"Sample count mismatch for {c_name}: expected {exp_cnt}, got {actual_counts[c_name]}"

    assert len(dataset) == 3669, f"Expected 3669 total test samples, got {len(dataset)}"

    # Load Model Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = get_resnet50_model(
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=False,
        dropout_rate=0.2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"\nSuccessfully loaded checkpoint from: {ckpt_path}")

    # Criterion for test loss calculation
    criterion = nn.CrossEntropyLoss()

    # Evaluate on held-out Test set
    print("\nEvaluating model on held-out TEST set...")
    test_loss, test_acc, y_true, y_pred = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    assert len(y_pred) == 3669, f"Expected 3669 predictions, got {len(y_pred)}"
    assert len(y_true) == 3669, f"Expected 3669 ground truth targets, got {len(y_true)}"

    # Metrics calculation
    labels = list(range(len(class_names)))

    # Per-class precision, recall, F1, support
    prec_cls, rec_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Macro & Weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n" + "=" * 60)
    print("FINAL RESNET50 HELD-OUT TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss:            {test_loss:.4f}")
    print(f"Test Accuracy:        {test_acc * 100:.2f}% ({test_acc:.4f})")
    print(f"Macro Precision:      {macro_p * 100:.2f}% ({macro_p:.4f})")
    print(f"Macro Recall:         {macro_r * 100:.2f}% ({macro_r:.4f})")
    print(f"Macro F1-Score:       {macro_f1 * 100:.2f}% ({macro_f1:.4f})")
    print(f"Weighted Precision:   {weighted_p * 100:.2f}% ({weighted_p:.4f})")
    print(f"Weighted Recall:      {weighted_r * 100:.2f}% ({weighted_r:.4f})")
    print(f"Weighted F1-Score:    {weighted_f1 * 100:.2f}% ({weighted_f1:.4f})")
    print("-" * 60)
    print("Per-Class Results:")
    for idx, c_name in enumerate(class_names):
        print(
            f"  {c_name:5s} -> Precision: {prec_cls[idx]*100:6.2f}% | "
            f"Recall: {rec_cls[idx]*100:6.2f}% | "
            f"F1: {f1_cls[idx]*100:6.2f}% | "
            f"Support: {supp_cls[idx]:4d}"
        )
    print("=" * 60)

    # Output directory setup
    out_tables_dir = Path(output_dir) / "tables" / "resnet50"
    out_figures_dir = Path(output_dir) / "figures" / "resnet50"
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save test_metrics.csv (per-class summary table)
    df_metrics = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": prec_cls,
            "Recall": rec_cls,
            "F1-Score": f1_cls,
            "Support": supp_cls,
        }
    )
    test_metrics_path = out_tables_dir / "test_metrics.csv"
    df_metrics.to_csv(test_metrics_path, index=False)
    print(f"\nSaved test metrics table to: {test_metrics_path}")

    # 2. Save test_classification_report.csv
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df_report = pd.DataFrame(report_dict).transpose()
    test_report_path = out_tables_dir / "test_classification_report.csv"
    df_report.to_csv(test_report_path)
    print(f"Saved classification report CSV to: {test_report_path}")

    # 3. Save test_summary.json
    summary_data = {
        "model": "ResNet50",
        "checkpoint": str(ckpt_path),
        "split": "test",
        "seed": seed,
        "num_test_samples": int(len(y_true)),
        "class_names": class_names,
        "per_class_sample_counts": actual_counts,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "evaluation_device": str(device),
        "evaluation_timestamp": datetime.now().isoformat(),
    }
    test_summary_path = out_tables_dir / "test_summary.json"
    with open(test_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved test summary JSON to: {test_summary_path}")

    # 4. Save test_confusion_matrix.png
    test_cm_path = out_figures_dir / "test_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=str(test_cm_path),
        title="ResNet50 Test Set Confusion Matrix",
    )
    plt.close("all")
    print(f"Saved test confusion matrix figure to: {test_cm_path}")

    print("\nHELD-OUT TEST EVALUATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 on Held-Out Test Set")
    parser.add_argument("--data-dir", type=str, default="data/classification", help="Path to classification dataset")
    parser.add_argument("--checkpoint", type=str, default="results/models/resnet50/best_model.pth", help="Checkpoint path")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")

    args = parser.parse_args()

    run_test_evaluation(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        seed=args.seed,
        num_workers=args.num_workers,
    )

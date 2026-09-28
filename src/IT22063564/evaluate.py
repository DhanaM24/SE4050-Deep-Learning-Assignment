"""Final evaluation of a trained model on train / val / test splits.

    python src/IT22063564/evaluate.py     (defaults to configs/mobilenetv2.yaml)

Only run this after all tuning is finished - the test split is used here and nowhere else.

Outputs (results/):
    tables/<model>_metrics.json            all metrics for every split + efficiency numbers
    tables/<model>_classification_report.csv   per-class precision/recall/F1 on test
    tables/model_comparison.csv            one row per model (shared across the group)
    figures/<model>_confusion_matrix.png
    figures/<model>_roc_curves.png
    figures/<model>_misclassified.png      sample test errors for the error analysis
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from dataset import labels_of, load_split  # noqa: E402
from utils import CONFIG_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR, ensure_dirs, load_config, load_json, save_json, set_seed  # noqa: E402


def compute_metrics(y_true, probs, n_classes):
    y_pred = probs.argmax(axis=1)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    y_onehot = np.eye(n_classes)[y_true]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_mac, "recall_macro": r_mac, "f1_macro": f_mac,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "roc_auc_macro_ovr": roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr"),
        "n_samples": int(len(y_true)),
    }


def measure_latency(model, img_size, n_runs=100):
    """Mean single-image inference time in ms (batch size 1, after warm-up)."""
    x = tf.random.uniform((1, img_size, img_size, 3), 0, 255)
    for _ in range(10):
        model(x, training=False)
    start = time.perf_counter()
    for _ in range(n_runs):
        model(x, training=False)
    return (time.perf_counter() - start) / n_runs * 1000


def plot_confusion(cm, classes, title, path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="True", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, probs, classes, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true == i, probs[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC {roc_auc_score(y_true == i, probs[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(test_ds, y_true, probs, classes, path, n=16):
    y_pred = probs.argmax(axis=1)
    wrong = np.flatnonzero(y_pred != y_true)[:n]
    if len(wrong) == 0:
        return
    images = np.concatenate([x.numpy() for x, _ in test_ds])[wrong]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
    for ax, img, idx in zip(np.ravel(axes), images, wrong):
        ax.imshow(img.astype("uint8"))
        ax.set_title(f"true {classes[y_true[idx]]} / pred {classes[y_pred[idx]]}\n"
                     f"p={probs[idx, y_pred[idx]]:.2f}", fontsize=8)
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def update_comparison_table(row):
    path = TABLES_DIR / "model_comparison.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not table.empty:
        table = table[table["model"] != row["model"]]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    print("\n", table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_DIR / "mobilenetv2.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    name, classes = cfg["model"], cfg["data"]["classes"]
    set_seed(cfg["seed"])
    ensure_dirs()

    model_path = MODELS_DIR / f"{name}.keras"
    model = tf.keras.models.load_model(model_path)

    results = {}
    for split in ("train", "val", "test"):
        ds = load_split(cfg, split, shuffle=False)
        y_true = labels_of(ds)
        probs = model.predict(ds, verbose=0)
        results[split] = compute_metrics(y_true, probs, len(classes))
        print(f"{split:>5}: " + ", ".join(f"{k}={v:.4f}" for k, v in results[split].items() if k != "n_samples"))

    # test-set artefacts (y_true / probs / ds still hold the test split)
    cm = confusion_matrix(y_true, probs.argmax(axis=1))
    results["test"]["confusion_matrix"] = cm.tolist()
    report = classification_report(y_true, probs.argmax(axis=1), target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(TABLES_DIR / f"{name}_classification_report.csv")
    plot_confusion(cm, classes, f"{name} - test confusion matrix", FIGURES_DIR / f"{name}_confusion_matrix.png")
    plot_roc(y_true, probs, classes, f"{name} - test ROC (one-vs-rest)", FIGURES_DIR / f"{name}_roc_curves.png")
    plot_misclassified(ds, y_true, probs, classes, FIGURES_DIR / f"{name}_misclassified.png")

    efficiency = {
        "total_params": int(model.count_params()),
        "model_size_mb": os.path.getsize(model_path) / 1e6,
        "latency_ms_per_image": measure_latency(model, cfg["train"]["img_size"]),
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }
    results["efficiency"] = efficiency
    save_json(results, TABLES_DIR / f"{name}_metrics.json")

    info_path = TABLES_DIR / f"{name}_train_info.json"
    info = load_json(info_path) if info_path.exists() else {}
    update_comparison_table({
        "model": name,
        "test_accuracy": results["test"]["accuracy"],
        "test_precision_macro": results["test"]["precision_macro"],
        "test_recall_macro": results["test"]["recall_macro"],
        "test_f1_macro": results["test"]["f1_macro"],
        "test_roc_auc_macro": results["test"]["roc_auc_macro_ovr"],
        "train_accuracy": results["train"]["accuracy"],
        "val_accuracy": results["val"]["accuracy"],
        "generalization_gap": results["train"]["accuracy"] - results["test"]["accuracy"],
        "params_millions": efficiency["total_params"] / 1e6,
        "model_size_mb": efficiency["model_size_mb"],
        "latency_ms": efficiency["latency_ms_per_image"],
        "train_minutes": info.get("total_train_seconds", float("nan")) / 60,
        "epochs_run": info.get("epochs_run"),
        "device": efficiency["device"],
    })


if __name__ == "__main__":
    main()

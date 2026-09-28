"""Train one model with two-phase transfer learning.

    python src/IT22063564/train.py        (defaults to configs/mobilenetv2.yaml)

Phase 1 trains only the new classification head on a frozen backbone.
Phase 2 unfreezes the top of the backbone and fine-tunes with a much lower LR.
Model selection uses the VALIDATION split only; the test split is never touched here.

Outputs (results/):
    models/<model>.keras              best checkpoint (lowest val_loss)
    tables/<model>_history.csv        per-epoch loss/accuracy for both phases
    tables/<model>_train_info.json    timing, parameter counts, config snapshot
    figures/<model>_learning_curves.png
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import pandas as pd
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from dataset import class_weights, load_split  # noqa: E402
from models import get_model_module  # noqa: E402
from utils import CONFIG_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR, ensure_dirs, load_config, save_json, set_seed  # noqa: E402


class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._start)


def callbacks(cfg, checkpoint_path, best_so_far=None):
    patience = cfg["train"]["early_stopping_patience"]
    return [
        # best_so_far stops phase 2 overwriting a better phase-1 checkpoint
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True,
                                           initial_value_threshold=best_so_far),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )


def count_params(model):
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    total = sum(int(tf.size(w)) for w in model.weights)
    return trainable, total


def run_phase(model, name, phase_cfg, cfg, train_ds, val_ds, weights, checkpoint_path, initial_epoch=0, best_so_far=None):
    compile_model(model, phase_cfg["learning_rate"])
    trainable, total = count_params(model)
    print(f"\n=== {name}: {trainable:,} trainable / {total:,} total params ===")
    timer = EpochTimer()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=initial_epoch + phase_cfg["epochs"], initial_epoch=initial_epoch,
        class_weight=weights, callbacks=callbacks(cfg, checkpoint_path, best_so_far) + [timer], verbose=2,
    )
    hist = pd.DataFrame(history.history)
    hist["epoch"] = range(initial_epoch + 1, initial_epoch + len(hist) + 1)
    hist["phase"] = name
    hist["epoch_seconds"] = timer.times
    return hist, trainable


def plot_learning_curves(hist, model_name, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(hist["epoch"], hist[metric], label="train")
        ax.plot(hist["epoch"], hist[f"val_{metric}"], label="validation")
        phase2 = hist[hist["phase"] == "phase2"]
        if not phase2.empty:
            ax.axvline(phase2["epoch"].iloc[0] - 0.5, color="grey", ls="--", label="start fine-tuning")
        ax.set(title=f"{model_name} - {metric}", xlabel="epoch", ylabel=metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_DIR / "mobilenetv2.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["model"]
    set_seed(cfg["seed"])
    ensure_dirs()

    train_ds, val_ds = load_split(cfg, "train"), load_split(cfg, "val")
    weights = class_weights(cfg) if cfg["train"]["use_class_weights"] else None
    print("Class weights:", weights)

    module = get_model_module(name)
    model = module.build_model(cfg)
    model.summary(expand_nested=False)
    checkpoint_path = str(MODELS_DIR / f"{name}.keras")

    start = time.perf_counter()
    hist1, trainable1 = run_phase(model, "phase1", cfg["phase1"], cfg, train_ds, val_ds, weights, checkpoint_path)
    histories, trainable2 = [hist1], None
    if cfg.get("phase2") and hasattr(module, "unfreeze_top"):
        module.unfreeze_top(model, cfg["phase2"]["unfreeze_from_layer"])
        hist2, trainable2 = run_phase(model, "phase2", cfg["phase2"], cfg, train_ds, val_ds, weights,
                                      checkpoint_path, initial_epoch=int(hist1["epoch"].iloc[-1]),
                                      best_so_far=float(hist1["val_loss"].min()))
        histories.append(hist2)
    total_seconds = time.perf_counter() - start

    hist = pd.concat(histories, ignore_index=True)
    hist.to_csv(TABLES_DIR / f"{name}_history.csv", index=False)
    plot_learning_curves(hist, name, FIGURES_DIR / f"{name}_learning_curves.png")

    best = hist.loc[hist["val_loss"].idxmin()]
    save_json({
        "model": name,
        "total_params": count_params(model)[1],
        "trainable_params_phase1": trainable1,
        "trainable_params_phase2": trainable2,
        "epochs_run": len(hist),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": best["val_loss"],
        "best_val_accuracy": best["val_accuracy"],
        "train_accuracy_at_best": best["accuracy"],
        "total_train_seconds": total_seconds,
        "mean_epoch_seconds": hist["epoch_seconds"].mean(),
        "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
        "tensorflow_version": tf.__version__,
        "config": cfg,
    }, TABLES_DIR / f"{name}_train_info.json")
    print(f"\nBest val_loss {best['val_loss']:.4f} (val_acc {best['val_accuracy']:.4f}) at epoch "
          f"{int(best['epoch'])}; training took {total_seconds / 60:.1f} min. Saved to {checkpoint_path}")


if __name__ == "__main__":
    main()

"""tf.data pipelines over the crops written by prepare_data.py.

Images are returned as raw float32 pixels in [0, 255]; each model does its own
normalisation inside the network, so every model reads exactly the same data.
"""

import numpy as np
import tensorflow as tf

from utils import PROJECT_ROOT


def load_split(cfg, split, shuffle=None):
    """Batched dataset for 'train', 'val' or 'test' with one-hot labels."""
    directory = PROJECT_ROOT / cfg["data"]["processed_dir"] / "crops" / split
    size = cfg["train"]["img_size"]
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        class_names=cfg["data"]["classes"],   # fixes the class -> index mapping
        label_mode="categorical",
        image_size=(size, size),
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split == "train") if shuffle is None else shuffle,
        seed=cfg["seed"],
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights(cfg):
    """'Balanced' class weights from the training split: n_samples / (n_classes * n_c)."""
    train_dir = PROJECT_ROOT / cfg["data"]["processed_dir"] / "crops" / "train"
    counts = np.array([len(list((train_dir / c).glob("*.jpg"))) for c in cfg["data"]["classes"]])
    weights = counts.sum() / (len(counts) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def labels_of(ds):
    """Integer ground-truth labels of an unshuffled dataset."""
    return np.concatenate([np.argmax(y, axis=1) for _, y in ds])

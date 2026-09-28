"""Shared helpers: config loading, seeding and small I/O utilities."""

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/IT22063564/utils.py -> repo root
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"


def load_config(model_config_path=None):
    """Load src/IT22063564/configs/common.yaml and optionally merge a model-specific config on top."""
    with open(CONFIG_DIR / "common.yaml") as f:
        cfg = yaml.safe_load(f)
    if model_config_path:
        with open(model_config_path) as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def set_seed(seed):
    """Seed every RNG we use so runs are repeatable (GPU kernels may still add tiny noise)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def ensure_dirs():
    for d in (FIGURES_DIR, MODELS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)

"""Central configuration loaded from the project .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Config:
    """All experiment settings in one place (no hard-coded values elsewhere)."""

    data_dir: Path = PROJECT_ROOT / _get("DATA_DIR", "dataset")
    train_dir: Path = PROJECT_ROOT / _get("TRAIN_DIR", "dataset/train")
    results_dir: Path = PROJECT_ROOT / _get("RESULTS_DIR", "results")

    img_size: int = int(_get("IMG_SIZE", "224"))
    batch_size: int = int(_get("BATCH_SIZE", "64"))
    num_classes: int = int(_get("NUM_CLASSES", "4"))
    class_names: list = _get("CLASS_NAMES", "D00,D10,D20,D40").split(",")

    seed: int = int(_get("SEED", "42"))
    test_size: float = float(_get("TEST_SIZE", "0.15"))
    val_size: float = float(_get("VAL_SIZE", "0.15"))

    head_epochs: int = int(_get("HEAD_EPOCHS", "100"))
    head_lr: float = float(_get("HEAD_LR", "0.001"))
    dropout: float = float(_get("DROPOUT", "0.5"))
    patience_earlystop: int = int(_get("PATIENCE_EARLYSTOP", "12"))
    patience_lr: int = int(_get("PATIENCE_LR", "5"))

    checkpoint_path: Path = PROJECT_ROOT / _get(
        "CHECKPOINT_PATH", "results/models/vgg16_multilabel.keras"
    )
    features_cache: Path = PROJECT_ROOT / _get(
        "FEATURES_CACHE", "results/models/vgg16_features.npz"
    )
    history_path: Path = PROJECT_ROOT / _get(
        "HISTORY_PATH", "results/metrics/training_history.json"
    )

    @classmethod
    def ensure_dirs(cls) -> None:
        for sub in (
            "figures",
            "metrics",
            "models",
            "predictions",
            "error_analysis",
            "tables",
        ):
            (cls.results_dir / sub).mkdir(parents=True, exist_ok=True)

    @classmethod
    def as_dict(cls) -> dict:
        return {
            "data_dir": str(cls.data_dir),
            "img_size": cls.img_size,
            "batch_size": cls.batch_size,
            "num_classes": cls.num_classes,
            "class_names": cls.class_names,
            "seed": cls.seed,
            "test_size": cls.test_size,
            "val_size": cls.val_size,
            "head_epochs": cls.head_epochs,
            "head_lr": cls.head_lr,
            "dropout": cls.dropout,
            "patience_earlystop": cls.patience_earlystop,
            "patience_lr": cls.patience_lr,
            "checkpoint_path": str(cls.checkpoint_path),
            "features_cache": str(cls.features_cache),
        }


Config.ensure_dirs()

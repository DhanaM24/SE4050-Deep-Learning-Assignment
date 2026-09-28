"""Model registry used by train.py / evaluate.py.

To add a model, create models/<name>.py exposing `build_model(cfg)` (and optionally
`unfreeze_top(model, from_layer)` for two-phase fine-tuning), then register it here.
"""

from importlib import import_module

MODEL_MODULES = {
    "mobilenetv2": "models.mobilenetv2",
}


def get_model_module(name):
    if name not in MODEL_MODULES:
        raise ValueError(f"Unknown model '{name}'. Registered: {list(MODEL_MODULES)}")
    return import_module(MODEL_MODULES[name])

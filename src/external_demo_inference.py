# -*- coding: utf-8 -*-
"""External Unseen Image Demonstration - ILLUSTRATIVE ONLY.

Downloads/runs the four trained project models on ONE image that is not part of
any evaluation pipeline. This script:

  * does NOT read, write or modify any train/val/test split,
  * does NOT influence thresholds, metrics or model weights,
  * exists purely to produce a qualitative inference example for the report.

The formal test-set results in results/<reg>/metrics/ remain untouched.

Usage (repo root):  venv\\Scripts\\python src\\external_demo_inference.py
"""
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as rn_pre
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre

ROOT = Path(__file__).resolve().parents[1]
IMG_PATH = ROOT / "results" / "external_demo" / "external_image.jpg"
OUT_PATH = ROOT / "results" / "external_demo" / "demo_predictions.json"
URL = ("https://images.squarespace-cdn.com/content/v1/573365789f726693272dc91a/"
       "1704992146415-CI272VYXPALWT52IGLUB/AdobeStock_201419293.jpeg?format=1500w")

DAMAGE_CLASSES = ["D00", "D10", "D20", "D40"]


def load_array():
    im = Image.open(IMG_PATH).convert("RGB").resize((224, 224))
    return np.asarray(im, dtype=np.float32)          # raw [0, 255]


def predict_multi_label(model_path, prep, thresholds_path, tag):
    x = load_array()[None, ...]
    x = prep(x)                                       # backbone-specific scaling
    model = tf.keras.models.load_model(model_path, compile=False)
    probs = model.predict(x, verbose=0)[0]
    thr = json.load(open(thresholds_path, encoding="utf-8"))
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return {
        "preprocessing": tag,
        "probabilities": {c: round(float(p), 4)
                          for c, p in zip(DAMAGE_CLASSES, probs)},
        "tuned_thresholds": {c: round(float(thr[c]), 4) for c in DAMAGE_CLASSES},
        "predicted_at_tuned_thresholds": [c for c, p in zip(DAMAGE_CLASSES, probs)
                                          if p >= thr[c]],
        "predicted_at_0_5": [c for c, p in zip(DAMAGE_CLASSES, probs) if p >= 0.5],
    }


def predict_softmax(model_path, classes, tag):
    x = load_array()[None, ...]                       # model scales internally
    model = tf.keras.models.load_model(model_path, compile=False)
    probs = model.predict(x, verbose=0)[0]
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    order = probs.argsort()[::-1]
    return {
        "preprocessing": tag,
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "top1": classes[order[0]],
        "top1_confidence": round(float(probs[order[0]]), 4),
    }


def main():
    results = {
        "purpose": "External unseen-image demonstration - NOT part of the formal "
                   "evaluation. No metric in results/<reg>/metrics/ is affected.",
        "image_url": URL,
        "image_path": str(IMG_PATH.relative_to(ROOT)),
        "image_size": list(Image.open(IMG_PATH).size),
        "external_to_rdd2020": "unverified - served from an AdobeStock image URL; "
                               "cannot be programmatically confirmed absent from "
                               "RDD2020",
        "predictions": {},
    }
    print("VGG16 ...")
    results["predictions"]["vgg16"] = predict_multi_label(
        ROOT / "src/IT22064868/models/vgg16_multilabel.keras",
        vgg_pre,
        ROOT / "results/IT22064868/metrics/thresholds.json",
        "resize 224x224 -> vgg16.preprocess_input (RGB->BGR, ImageNet mean "
        "subtraction) -> frozen backbone + head")
    print("ResNet-50 ...")
    results["predictions"]["resnet50"] = predict_multi_label(
        ROOT / "src/IT23325814/models/resnet50_multilabel.keras",
        rn_pre,
        ROOT / "results/IT23325814/metrics/thresholds.json",
        "resize 224x224 -> resnet50.preprocess_input (ImageNet mean subtraction) "
        "-> frozen backbone + head")
    print("MobileNetV2 ...")
    results["predictions"]["mobilenetv2"] = predict_softmax(
        ROOT / "results/IT22063564/models/mobilenetv2.keras",
        DAMAGE_CLASSES,
        "resize 224x224, raw [0,255] pixels (Rescaling to [-1,1] inside the "
        "model); NOTE: model was trained on box crops, whole image used here")
    print("Custom CNN ...")
    results["predictions"]["custom_cnn"] = predict_multi_label(
        ROOT / "results/IT23331518/custom_cnn_model.keras",
        lambda a: a,                                  # raw [0,255]; Rescaling in model
        ROOT / "results/IT23331518/metrics/thresholds.json",
        "resize 224x224, raw [0,255] pixels (Rescaling 1/255 inside the model)")

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["predictions"], indent=2))
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()

"""MobileNetV2 classifier for road-damage crops.

Why MobileNetV2 (Sandler et al., 2018):
  * depthwise-separable convolutions + inverted residuals with linear bottlenecks
    give ImageNet-level features with ~3.5M parameters (VGG16 ~138M, ResNet50 ~25M);
  * it is designed for mobile/edge inference, which matches the real use case of
    road inspection from a phone or dash-cam;
  * it is the efficiency end of our accuracy-vs-cost comparison.

Architecture:
  input 224x224x3 (0-255)
  -> augmentation (train only: flip, small rotation, zoom, contrast)
  -> Rescaling to [-1, 1]            (same as mobilenet_v2.preprocess_input)
  -> MobileNetV2 backbone, ImageNet weights, no top  -> 7x7x1280
  -> GlobalAveragePooling2D                          -> 1280
  -> Dropout -> Dense(128, ReLU, L2) -> Dropout
  -> Dense(4, softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers

BACKBONE_PREFIX = "mobilenetv2_"  # Keras names the backbone e.g. "mobilenetv2_1.00_224"


def build_augmentation(aug_cfg):
    steps = []
    if aug_cfg.get("horizontal_flip"):
        steps.append(layers.RandomFlip("horizontal"))
    if aug_cfg.get("rotation"):
        steps.append(layers.RandomRotation(aug_cfg["rotation"]))
    if aug_cfg.get("zoom"):
        steps.append(layers.RandomZoom(aug_cfg["zoom"]))
    if aug_cfg.get("contrast"):
        steps.append(layers.RandomContrast(aug_cfg["contrast"]))
    return tf.keras.Sequential(steps, name="augmentation")


def build_model(cfg):
    arch = cfg["architecture"]
    size = cfg["train"]["img_size"]
    n_classes = len(cfg["data"]["classes"])

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False,
        weights=arch["weights"], alpha=arch["alpha"])
    backbone.trainable = False  # phase 1: feature extractor only

    inputs = layers.Input(shape=(size, size, 3), name="image")
    x = build_augmentation(cfg["augmentation"])(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="to_minus1_1")(x)
    # training=False keeps BatchNorm statistics frozen even after unfreezing,
    # which stops the small dataset from wrecking the ImageNet statistics.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(arch["dropout"])(x)
    if arch["dense_units"]:
        x = layers.Dense(arch["dense_units"], activation="relu",
                         kernel_regularizer=regularizers.l2(arch["l2"]), name="head_dense")(x)
        x = layers.Dropout(arch["dropout"])(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_classifier")


def unfreeze_top(model, from_layer):
    """Phase 2: make backbone layers [from_layer:] trainable; BatchNorm layers stay frozen."""
    backbone = next(l for l in model.layers if l.name.startswith(BACKBONE_PREFIX))
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= from_layer and not isinstance(layer, layers.BatchNormalization)

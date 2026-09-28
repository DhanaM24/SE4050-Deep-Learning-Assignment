# SE4050-Deep-Learning-Assignment

Road-damage type classification on the **RDD2020** dataset, comparing four CNNs
under identical data splits and training conditions: a custom CNN, VGG16, ResNet50 and MobileNetV2.

## Dataset

- **RDD2020: Road Damage Dataset 2020** - Arya, D. et al., *Data in Brief* 36 (2021) 107133,
  doi:10.1016/j.dib.2021.107133. Licence: CC BY 4.0.
  Download: https://data.mendeley.com/datasets/5ty2wb6gvg/1 (or https://github.com/sekilab/RoadDamageDetector)
- Road photos from Czech Republic, India and Japan with Pascal-VOC bounding boxes.
- Classes used: `D00` longitudinal crack, `D10` transverse crack, `D20` alligator crack, `D40` pothole
  (the official label map). Other tags found in the XMLs (`D01`, `D11`, `D43`, `D44`, `D50`, ...) are dropped.

**Task framing.** Every bounding box is cropped (square window with 15% context) and classified into one of
the four damage types. The published `test1`/`test2` archives have **no annotations**, so our labelled
test set is carved out of `train.tar.gz`: 70/15/15 train/val/test, split **by source image** so crops from
the same photo never appear in two splits (checked by an assertion in `prepare_data.py`).

## Setup

```bash
pip install -r requirements.txt
```

Place `train.tar.gz` in `data/` and extract it:

```bash
mkdir -p data/raw && tar -xzf data/train.tar.gz -C data/raw     # -> data/raw/train/{Czech,India,Japan}
python src/IT22063564/prepare_data.py                             # -> data/processed/crops/{train,val,test}/<class>/
```

All shared settings (seed = 42, image size, batch size, split ratios, classes) live in
[src/IT22063564/configs/common.yaml](src/IT22063564/configs/common.yaml). Change them only as a group - every model must then be retrained.

## Train and evaluate a model

```bash
python src/IT22063564/train.py      # --config defaults to src/IT22063564/configs/mobilenetv2.yaml
python src/IT22063564/evaluate.py
```

`train.py` uses only train + validation data; `evaluate.py` is the only script that reads the test split.
Results are written to `results/`:

| File | Content |
|---|---|
| `tables/<model>_history.csv`, `figures/<model>_learning_curves.png` | per-epoch loss/accuracy |
| `tables/<model>_metrics.json` | accuracy, macro/weighted P/R/F1, ROC-AUC for train/val/test, latency, size |
| `tables/<model>_classification_report.csv` | per-class test metrics |
| `figures/<model>_confusion_matrix.png`, `_roc_curves.png`, `_misclassified.png` | test visualisations |
| `tables/model_comparison.csv` | one row per model - the shared comparison table |

### Running on Google Colab (GPU)

```python
!git clone <repo-url> && cd SE4050-Deep-Learning-Assignment
# upload/copy data/processed/crops (zip it locally after prepare_data.py) or train.tar.gz from Drive
!pip install -q -r requirements.txt
!python src/IT22063564/train.py && python src/IT22063564/evaluate.py
```

## Adding a model (for each member)

1. Create `src/IT22063564/models/<name>.py` with `build_model(cfg)` (and `unfreeze_top(model, from_layer)` if it
   uses two-phase fine-tuning). The model must take raw 0-255 pixels and do its own preprocessing inside
   the network - see [src/IT22063564/models/mobilenetv2.py](src/IT22063564/models/mobilenetv2.py).
2. Register it in [src/IT22063564/models/\_\_init\_\_.py](src/IT22063564/models/__init__.py).
3. Add `src/IT22063564/configs/<name>.yaml` (copy `mobilenetv2.yaml`).

## Models

| Model | Owner | Config |
|---|---|---|
| Custom CNN | | |
| VGG16 | | |
| ResNet50 | | |
| MobileNetV2 | IT22063564 | [src/IT22063564](src/IT22063564), notebook [notebooks/IT22063564/mobilenetv2_training.ipynb](notebooks/IT22063564/mobilenetv2_training.ipynb) |

### MobileNetV2

ImageNet-pretrained MobileNetV2 backbone (~2.3M params, inverted residuals + depthwise-separable
convolutions) with a GAP -> Dropout(0.3) -> Dense(128, ReLU, L2) -> Dropout(0.3) -> Dense(4, softmax) head.
Trained in two phases with Adam and class-weighted categorical cross-entropy:
1. backbone frozen, head only - lr 1e-3, up to 10 epochs;
2. backbone layers 100+ unfrozen (BatchNorm kept frozen) - lr 1e-5, up to 20 epochs.

Early stopping (patience 5) and ReduceLROnPlateau on validation loss; the checkpoint with the lowest
validation loss is kept. Augmentation: horizontal flip, rotation up to +-18 deg, zoom 10%, contrast 10%
(no 90 deg rotations - they would turn a longitudinal crack into a transverse one).

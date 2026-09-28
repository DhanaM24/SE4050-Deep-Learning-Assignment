# SE4050 Deep Learning 2026 — Multi-Label Road-Damage Classification (VGG16)

Multi-label classifier for the four road-damage classes in `label_map.pbtxt`
(**D00** longitudinal crack, **D10** transverse crack, **D20** alligator crack,
**D40** pothole) built as part of the group project. The whole
pipeline - validation, leakage-safe split, VGG16 training, evaluation and
inference - lives in one self-contained notebook,
`notebooks/VGG_16.ipynb`; the shared split manifest
(`results/metrics/split_manifest.csv`) is used by every group model.

Headline result (frozen ImageNet VGG16 + trained head, single seed 42):

| metric (test, 645 images) | @0.5 | validation-tuned thresholds |
|---|---|---|
| micro F1   | 0.692 | 0.696 |
| macro F1   | 0.556 | **0.649** |
| micro / macro ROC-AUC | 0.864 / 0.832 | — |
| subset accuracy | 0.481 | 0.389 |

Full numbers, figures and error analysis are produced by running the notebook;
metrics and figures go to `results/`, saved models to `src/models/`.

## Repository layout

```
dataset/                  # extracted dataset (git-ignored, see below)
notebooks/VGG_16.ipynb    # the whole pipeline: validation -> split -> model -> evaluation -> inference
src/models/               # trained model + head weights + run metadata (git-ignored)
results/cache/            # precomputed VGG16 feature caches (git-ignored)
results/                  # metrics json, figures, predictions, error analysis
.env.example              # environment template (copy to .env)
requirements.txt          # pinned dependencies
```

## Requirements

* Python **3.11** (tested on Windows 11; TensorFlow 2.21 has **no GPU support on
  native Windows**, so everything runs on CPU — the pipeline is designed for it:
  frozen backbone + precomputed features).
* ~1.5 GB free disk (dataset images + feature caches).

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env        # adjust paths only if needed
```

## Dataset access

The dataset is **not** committed (see `.gitignore`). Place the extracted
images so that the following exists (structure described in the original
`FileStructure.txt` shipped with the dataset):

```
dataset/
  train/Czech/{images,labels}/     # XML labels
  train/India/{images,labels}/     # XML labels
  test1/ test2/                    # images only, no damage annotations
  train/label_map.pbtxt            # D00/D10/D20/D40 class map
```

Notes recorded during validation (`results/metrics/dataset_stats.json`):

* 15,830 extracted images, all intact (0 corrupt, 0 exact duplicates). This is
  the original 18,437 minus the unused `train/Japan` extraction (2,607 images),
  which was removed locally: validation had reported its one corrupt file
  (`Japan_004643.jpg`) and the images cannot be supervised (see below).
* `train.tar.gz` is truncated upstream — the Japanese labels were never
  recovered, so supervised training uses **Czech + India only** (4,295 labeled
  images).
* `test1`/`test2` ship no damage annotations and are therefore description-only;
  all reported metrics use the held-out **test** split of the labeled set.

## Execution (in order)

```powershell
# 1. register the kernel once
python -m ipykernel install --user --name se4050-venv

# 2. run the notebook end-to-end (~10 min):
#    validates the dataset, builds the split, extracts features, trains,
#    evaluates, tunes thresholds, plots figures, classifies a sample image
python -m nbconvert --to notebook --execute --inplace notebooks\VGG_16.ipynb `
    --ExecutePreprocessor.kernel_name=se4050-venv --ExecutePreprocessor.timeout=2400
```

The notebook is self-contained (config, validation and split code are all
inside it): it re-validates, re-splits (`force=True`), extracts VGG16 features
with caching, trains the head, evaluates on the untouched test set, tunes
thresholds **on validation only**, performs the error analysis, and writes
metrics/figures to `results/` and the saved model to `src/models/`.

## Reproducibility

* Single global seed `42` (Python / NumPy / TensorFlow), set in the notebook
  setup section from `.env`.
* Split manifest: `results/metrics/split_manifest.csv` — shared source of
  truth for all group models; leakage report: `split_report.json`
  (0 clusters spanning splits, 0 duplicate paths).
* Test set is used **only** in the final evaluation cells; threshold tuning
  uses validation only (assignment requirement).
* TensorFlow oneDNN CPU kernels may cause tiny numeric variation between runs;
  bit-level reproducibility is not claimed.

## Key outputs

| artifact | contents |
|---|---|
| `results/metrics/test_metrics.json` | test metrics @0.5 (per-class P/R/F1/AUC) |
| `results/metrics/test_metrics_tuned.json` | same at validation-tuned thresholds |
| `results/metrics/thresholds.json` | tuned per-class thresholds |
| `results/metrics/run_summary.json` | config, timings, parameter counts |
| `results/figures/` | EDA, learning curves, ROC, confusion matrices |
| `results/error_analysis/` | FN/FP counts per class, confident-error grid |
| `src/models/vgg16_multilabel.keras` | full saved model (backbone + trained head) |
| `src/models/vgg16_head.weights.h5` | best head weights (validation AUC) |
# SE4050-Deep-Learning-Assignment
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

# SE4050 Deep Learning 2026 — Road-Damage Classification (Group Project)

Road-damage type classification on the **RDD2020** dataset, comparing four CNNs
under identical data, splits and training conditions:

| Model | Task | Owner | Notebook |
|---|---|---|---|
| Custom CNN | country-level image classification (demo baseline) | IT23331518 | [notebooks/IT23331518/custom_cnn.ipynb](notebooks/IT23331518/custom_cnn.ipynb) |
| VGG16 | multi-label damage classification (full images) | IT22064868 | [notebooks/IT22064868/VGG_16.ipynb](notebooks/IT22064868/VGG_16.ipynb) |
| ResNet50 | multi-label damage classification (full images) | IT23325814 | [notebooks/IT23325814/ResNet50.ipynb](notebooks/IT23325814/ResNet50.ipynb) |
| MobileNetV2 | multi-class damage classification (bounding-box crops) | IT22063564 | [notebooks/IT22063564/mobilenetv2_training.ipynb](notebooks/IT22063564/mobilenetv2_training.ipynb) |

Classes (official RDD2020 label map): **D00** longitudinal crack, **D10**
transverse crack, **D20** alligator crack, **D40** pothole. Other tags found in
the XMLs (`D01`, `D44`, …) are dropped.

## Dataset

- **RDD2020: Road Damage Dataset 2020** — Arya, D. et al., *Data in Brief* 36
  (2021) 107133, doi:10.1016/j.dib.2021.107133. Licence: CC BY 4.0.
  Download: https://data.mendeley.com/datasets/5ty2wb6gvg/1 (or
  https://github.com/sekilab/RoadDamageDetector)
- Road photos from Czech Republic, India and Japan with Pascal-VOC bounding
  boxes.

The dataset is **not** committed (see `.gitignore`). Place the extracted
archive so that this exists (structure described in `data/FileStructure.txt`):

```
data/
  train/Czech/{images,annotations/xmls}/   # 2,829 images, all annotated
  train/India/{images,annotations/xmls}/   # 7,706 images, all annotated
  train/Japan/images/                      # 2,607 images, no annotations here
  train/label_map.pbtxt                    # D00/D10/D20/D40 class map
  test1/  test2/                           # images only, no damage annotations
```

Validation facts (recorded by every notebook into `results/<reg>/metrics/`):

* 13,142 train images; 1 corrupt file (`Japan_004643.jpg`) is detected and
  skipped, leaving 13,141 usable images.
* 10,535 annotated train images (Czech + India); **4,295** carry at least one
  of the four target classes — these form the labelled multi-label set
  (1,300 of them have more than one damage type).
* `test1`/`test2` and the Japan train images ship without damage annotations
  and are description-only; all reported metrics use the held-out **test**
  split of the labelled set (or of the crops carved from it).

**Task framing.** VGG16 and ResNet50 classify the full image (multi-label,
sigmoid + BCE). MobileNetV2 classifies square crops around each bounding box
(15% context, 224×224, multi-class softmax); its labelled test set is carved
out of `train` **by source image** so crops from one photo never cross splits.
The custom CNN trains a small from-scratch CNN on the three country folders as
a baseline demo.

## Repository layout

```
notebooks/<reg>/            # one self-contained notebook per member
src/<reg>/                  # member scripts/models (VGG/ResNet keep everything in the notebook)
  IT22063564/               # MobileNetV2 pipeline (prepare_data/train/evaluate + configs)
  IT22064868/models/        # VGG16 checkpoints + run metadata (weights git-ignored)
  IT23325814/               # ResNet50 checkpoints (weights git-ignored) + helpers
  IT23331518/               # custom CNN helpers
results/<reg>/              # per-member metrics, figures, tables, caches
  IT22064868/cache/         # precomputed VGG16 features (git-ignored)
  tables/model_comparison.csv
data/                       # dataset (git-ignored, see above)
.env.example                # environment template (copy to .env)
requirements.txt            # dependencies
```

## Requirements

* Python **3.11** (tested on Windows 11). TensorFlow ≥ 2.11 has **no GPU
  support on native Windows**, so all notebooks run on CPU — the pipelines are
  designed for it (frozen backbones, precomputed features, early stopping).
* ~4 GB free disk (dataset + crops + feature caches + venv).

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt        # torch/torchvision are optional (PyTorch helpers)
Copy-Item .env.example .env            # VGG16 paths/hyper-parameters
python -m ipykernel install --user --name se4050-venv   # kernel used below
```

## Execution

Each notebook is self-contained (config, validation and split code live inside
it) and is executed with the project-root walk, so run from the repository
root:

```powershell
python -m nbconvert --to notebook --execute --inplace notebooks\IT22064868\VGG_16.ipynb `
    --ExecutePreprocessor.kernel_name=se4050-venv --ExecutePreprocessor.timeout=7200
python -m nbconvert --to notebook --execute --inplace notebooks\IT23325814\ResNet50.ipynb `
    --ExecutePreprocessor.kernel_name=se4050-venv --ExecutePreprocessor.timeout=7200
python -m nbconvert --to notebook --execute --inplace notebooks\IT23331518\custom_cnn.ipynb `
    --ExecutePreprocessor.kernel_name=se4050-venv --ExecutePreprocessor.timeout=7200
python -m nbconvert --to notebook --execute --inplace notebooks\IT22063564\mobilenetv2_training.ipynb `
    --ExecutePreprocessor.kernel_name=se4050-venv --ExecutePreprocessor.timeout=21600
```

Order does not matter; the MobileNetV2 run is the longest (crops are built on
the first run, then two training phases).

## Shared methodology

* Single global seed **42** (Python / NumPy / TensorFlow).
* VGG16 and ResNet50 use the **same labelled set and the same split
  algorithm** (64-bit dHash near-duplicate clusters assigned as whole units,
  70/15/15). ResNet50 re-computes the split and cross-checks it against the
  shared manifest `results/IT22064868/metrics/split_manifest.csv`
  (agreement is saved to `results/IT23325814/metrics/shared_manifest_check.json`).
* Threshold tuning happens on **validation only**; the test set is touched
  once, in the final evaluation cells (assignment rule).
* TensorFlow oneDNN CPU kernels may cause tiny numeric variation between runs;
  bit-level reproducibility is not claimed.

## Key outputs (per member)

| Artifact | Contents |
|---|---|
| `results/<reg>/metrics/test_metrics.json` | test metrics @0.5 (per-class P/R/F1/AUC) |
| `results/<reg>/metrics/test_metrics_tuned.json` | same at validation-tuned thresholds |
| `results/<reg>/metrics/thresholds.json` | tuned per-class thresholds |
| `results/<reg>/metrics/run_summary.json` | config, timings, parameter counts |
| `results/<reg>/figures/` | EDA, learning curves, ROC, confusion matrices |
| `results/<reg>/tables/` | per-class tables, histories, classification reports |
| `src/<reg>/models/*.keras` | saved models (git-ignored; share via Drive) |

MobileNetV2 additionally writes `tables/model_comparison.csv` (one row per
model — the shared comparison table) and `data/processed/splits.csv`
(per-crop split assignment, leakage guard asserted in `prepare_data.py`).

## Citation

Arya, D. et al. (2021). RDD2020: A multi-labeled image dataset for
general-purpose road damage detection. *Data in Brief*, 36, 107133.
https://doi.org/10.1016/j.dib.2021.107133

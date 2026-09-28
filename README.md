# SE4050 Deep Learning 2026 — Multi-Label Road-Damage Classification (VGG16)

Multi-label classifier for the four road-damage classes in `label_map.pbtxt`
(**D00** longitudinal crack, **D10** alligator crack, **D20** pothole,
**D40** transverse crack) built as part of the group project. This repository
contains the **VGG16 experiment** (`notebooks/VGG_16.ipynb`) plus the shared
data validation and leakage-safe split utilities used by every group model.

Headline result (frozen ImageNet VGG16 + trained head, single seed 42):

| metric (test, 645 images) | @0.5 | validation-tuned thresholds |
|---|---|---|
| micro F1   | 0.692 | 0.696 |
| macro F1   | 0.556 | **0.649** |
| micro / macro ROC-AUC | 0.864 / 0.832 | — |
| subset accuracy | 0.481 | 0.389 |

Full numbers, figures and error analysis are produced by running the notebook;
artifacts are written to `results/`.

## Repository layout

```
dataset/                  # extracted dataset (git-ignored, see below)
notebooks/VGG_16.ipynb    # main experiment: EDA -> model -> training -> evaluation
src/config.py             # central config loaded from .env
src/utils.py              # seeding, json helpers, timer, plotting style
src/validate_dataset.py   # integrity/duplicate scan + EDA figures
src/dataset.py            # cluster-grouped, leakage-safe train/val/test split
results/                  # metrics json, figures, predictions, model artifacts
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
# 1. validate every image + build EDA figures (~5 min)
python -m src.validate_dataset

# 2. build the leakage-safe split (grouped by near-duplicate cluster) (~1 min)
python -m src.dataset

# 3. register the kernel once, then run the notebook end-to-end (~10 min)
python -m ipykernel install --user --name se4050-venv
python -m nbconvert --to notebook --execute --inplace notebooks\VGG_16.ipynb `
    --ExecutePreprocessor.kernel_name=se4050-venv --ExecutePreprocessor.timeout=2400
```

The notebook itself is self-contained: it re-validates, re-splits
(`force=True`), extracts VGG16 features with caching, trains the head,
evaluates on the untouched test set, tunes thresholds **on validation only**,
performs the error analysis and writes all artifacts to `results/`.

## Reproducibility

* Single global seed `42` (Python / NumPy / TensorFlow) via `src/config.py`.
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

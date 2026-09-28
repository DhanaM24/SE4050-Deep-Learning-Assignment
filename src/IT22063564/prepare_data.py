"""Turn the RDD2020 detection annotations into a road-damage *classification* dataset.

Each bounding box becomes one square image crop labelled with its damage type
(D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole).

The RDD2020 test1/test2 archives ship without annotations, so our labelled test set
is carved out of the training archive. The split is done per source IMAGE (all crops
from one photo land in the same split) to prevent near-duplicate leakage between
train and test.

Usage:
    python src/IT22063564/prepare_data.py
Outputs:
    data/processed/crops/{train,val,test}/{D00,D10,D20,D40}/*.jpg
    data/processed/splits.csv            one row per crop
    data/processed/dataset_summary.json  counts used in the report's EDA section
"""

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, load_config, save_json  # noqa: E402


def parse_annotation(xml_path):
    """Return [(label, xmin, ymin, xmax, ymax), ...] for one VOC-style XML file."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        coords = [int(float(bb.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append((obj.find("name").text.strip(), *coords))
    return boxes


def square_context_box(xmin, ymin, xmax, ymax, img_w, img_h, context):
    """Square window centred on the box, enlarged by `context`, shifted to stay inside the image.

    A square window avoids stretching long thin cracks when resizing, and the extra
    context shows the surrounding road surface.
    """
    side = max(xmax - xmin, ymax - ymin) * (1 + 2 * context)
    side = int(min(side, img_w, img_h))
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    left = int(min(max(cx - side / 2, 0), img_w - side))
    top = int(min(max(cy - side / 2, 0), img_h - side))
    return left, top, left + side, top + side


def collect_boxes(cfg):
    data_cfg = cfg["data"]
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]
    classes = set(data_cfg["classes"])
    rows, dropped = [], Counter()

    for country in data_cfg["countries"]:
        xml_dir = raw_dir / country / "annotations" / "xmls"
        xml_files = sorted(xml_dir.glob("*.xml"))
        if not xml_files:
            print(f"WARNING: no annotations found for {country} in {xml_dir} - skipped")
            continue
        for xml_path in xml_files:
            image_path = raw_dir / country / "images" / f"{xml_path.stem}.jpg"
            if not image_path.exists():
                dropped["missing_image"] += 1
                continue
            for i, (label, xmin, ymin, xmax, ymax) in enumerate(parse_annotation(xml_path)):
                if label not in classes:
                    dropped[f"label_{label}"] += 1
                    continue
                if min(xmax - xmin, ymax - ymin) < data_cfg["min_box_side"]:
                    dropped["too_small"] += 1
                    continue
                rows.append({
                    "image_id": xml_path.stem, "country": country, "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                    "box_idx": i, "label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                })
    return pd.DataFrame(rows), dropped


def split_by_image(df, split_cfg, seed):
    """Stratified split on images; strata = country + rarest class present in the image."""
    class_freq = df["label"].value_counts()
    per_image = df.groupby("image_id").agg(country=("country", "first"), labels=("label", set))
    per_image["stratum"] = per_image.apply(
        lambda r: f"{r.country}_{min(r.labels, key=lambda c: class_freq[c])}", axis=1)
    # strata that are too small to split fall back to the country alone
    small = per_image["stratum"].map(per_image["stratum"].value_counts()) < 10
    per_image.loc[small, "stratum"] = per_image.loc[small, "country"]

    holdout = split_cfg["val"] + split_cfg["test"]
    train_ids, rest_ids = train_test_split(
        per_image.index, test_size=holdout, stratify=per_image["stratum"], random_state=seed)
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=split_cfg["test"] / holdout,
        stratify=per_image.loc[rest_ids, "stratum"], random_state=seed)

    split_of = {**{i: "train" for i in train_ids}, **{i: "val" for i in val_ids},
                **{i: "test" for i in test_ids}}
    return df["image_id"].map(split_of)


def save_crops(df, cfg):
    data_cfg = cfg["data"]
    out_root = PROJECT_ROOT / data_cfg["processed_dir"] / "crops"
    size = data_cfg["crop_size"]
    crop_paths = []
    for image_path, group in df.groupby("image_path", sort=False):
        with Image.open(PROJECT_ROOT / image_path) as img:
            img = img.convert("RGB")
            for idx, r in group.iterrows():
                box = square_context_box(r.xmin, r.ymin, r.xmax, r.ymax, *img.size, data_cfg["crop_context"])
                out_path = out_root / r.split / r.label / f"{r.image_id}_{r.box_idx}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.crop(box).resize((size, size), Image.BICUBIC).save(out_path, quality=95)
                crop_paths.append((idx, out_path.relative_to(PROJECT_ROOT).as_posix()))
    return pd.Series(dict(crop_paths))


def main():
    cfg = load_config()
    df, dropped = collect_boxes(cfg)
    if df.empty:
        sys.exit("No boxes found - extract data/train.tar.gz into data/raw first (see README).")

    df["split"] = split_by_image(df, cfg["data"]["split"], cfg["seed"])
    df["crop_path"] = save_crops(df, cfg)

    out_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    df.to_csv(out_dir / "splits.csv", index=False)

    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    summary = {
        "n_images": int(df["image_id"].nunique()),
        "n_crops": len(df),
        "countries": df.groupby("country")["image_id"].nunique().to_dict(),
        "crops_per_split_and_class": counts.to_dict(orient="index"),
        "images_per_split": df.groupby("split")["image_id"].nunique().to_dict(),
        "dropped": dict(dropped),
    }
    save_json(summary, out_dir / "dataset_summary.json")

    # leakage guard: no source image may appear in more than one split
    assert df.groupby("image_id")["split"].nunique().max() == 1
    print(counts, "\n", f"Saved {len(df)} crops from {summary['n_images']} images to {out_dir}")
    print("Dropped:", dict(dropped))


if __name__ == "__main__":
    main()

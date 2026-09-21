"""Dataset validation and EDA.

Checks the extracted RDD-style dataset for structure, class distribution,
image integrity, duplicates and label availability, then saves statistics
and figures under results/ (no data is ever modified here).

Run:  python -m src.validate_dataset
"""

import collections
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import Config
from src.utils import apply_style, save_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_images() -> pd.DataFrame:
    """Walk every split/class folder and collect one row per image."""
    rows = []
    for split in ("train", "test1", "test2"):
        split_dir = Config.data_dir / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            images_dir = class_dir / "images"
            if not images_dir.is_dir():
                continue
            for img in sorted(images_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTS:
                    rows.append(
                        {
                            "split": split,
                            "source": class_dir.name,
                            "path": img,
                            "bytes": img.stat().st_size,
                        }
                    )
    return pd.DataFrame(rows)


def check_images(df: pd.DataFrame) -> tuple:
    """Open every image; return per-image status and size info."""
    statuses, widths, heights, corrupt = [], [], [], []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                widths.append(im.width)
                heights.append(im.height)
            statuses.append("ok")
        except Exception as exc:  # noqa: BLE001 - report every failure mode
            statuses.append("corrupt")
            widths.append(np.nan)
            heights.append(np.nan)
            corrupt.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    return statuses, widths, heights, corrupt


def exact_duplicates(df: pd.DataFrame) -> list:
    """Find byte-identical images via content MD5."""
    by_hash = collections.defaultdict(list)
    for p in df["path"]:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        by_hash[h].append(str(p))
    return [v for v in by_hash.values() if len(v) > 1]


def parse_damage_labels() -> pd.DataFrame:
    """Build the multi-label annotation table from train VOC xml files."""
    MAIN4 = Config.class_names
    rows = []
    for country in ("Czech", "India"):
        xml_dir = Config.train_dir / country / "annotations" / "xmls"
        if not xml_dir.is_dir():
            continue
        for xml_path in sorted(xml_dir.glob("*.xml")):
            root = ET.parse(xml_path).getroot()
            objects = root.findall("object")
            classes = sorted(
                {o.findtext("name") for o in objects if o.findtext("name") in MAIN4}
            )
            img = xml_path.parents[2] / "images" / (xml_path.stem + ".jpg")
            if not img.exists():
                continue
            row = {"path": str(img), "source": country, "n_objects": len(objects)}
            for c in MAIN4:
                row[c] = int(c in classes)
            row["labels"] = classes
            rows.append(row)
    return pd.DataFrame(rows)


def label_statistics(labels_df: pd.DataFrame) -> dict:
    stats = {
        "annotated_images": int(len(labels_df)),
        "images_with_no_object": int((labels_df["n_objects"] == 0).sum()),
        "images_with_main4_label": int(labels_df[Config.class_names].sum(axis=1).gt(0).sum()),
        "images_without_main4_label": int(labels_df[Config.class_names].sum(axis=1).eq(0).sum()),
        "per_class_image_counts": {
            c: int(labels_df[c].sum()) for c in Config.class_names
        },
        "multi_label_images": int((labels_df[Config.class_names].sum(axis=1) > 1).sum()),
        "single_label_images": int((labels_df[Config.class_names].sum(axis=1) == 1).sum()),
        "per_source": labels_df["source"].value_counts().to_dict(),
    }
    per_source_class = {}
    for src, grp in labels_df.groupby("source"):
        per_source_class[src] = {c: int(grp[c].sum()) for c in Config.class_names}
    stats["per_source_class_counts"] = per_source_class
    return stats


def make_figures(df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    apply_style()
    fig_dir = Config.results_dir / "figures"

    # 1. images per split/source
    counts = df.groupby(["split", "source"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", figsize=(8, 4), rot=0, colormap="tab10")
    ax.set_title("Images per split and source country")
    ax.set_ylabel("number of images")
    ax.legend(title="source")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution_splits.png")
    plt.close()

    # 2. damage-class distribution (multi-label aware)
    per_class = labels_df[Config.class_names].sum()
    ax = per_class.plot(kind="bar", figsize=(7, 4), rot=0, color="#4c72b0")
    ax.set_title("Annotated images per damage class (train/Czech+India)")
    ax.set_ylabel("images containing class")
    for i, v in enumerate(per_class):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(fig_dir / "damage_class_distribution.png")
    plt.close()

    # 3. labels per image
    n_labels = labels_df[Config.class_names].sum(axis=1)
    ax = n_labels.value_counts().sort_index().plot(kind="bar", figsize=(6, 4), rot=0)
    ax.set_title("Number of damage classes per annotated image")
    ax.set_xlabel("labels per image")
    ax.set_ylabel("images")
    plt.tight_layout()
    plt.savefig(fig_dir / "labels_per_image.png")
    plt.close()

    # 4. per-source class composition
    src_cls = pd.DataFrame(stats_global["per_source_class_counts"]).T
    ax = src_cls.plot(kind="bar", figsize=(7, 4), rot=0, colormap="Set2")
    ax.set_title("Class distribution by source country")
    ax.set_ylabel("images containing class")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution_by_source.png")
    plt.close()


stats_global = {}


def main() -> dict:
    print("Scanning dataset ...")
    df = scan_images()
    print(f"found {len(df)} images")

    print("Checking image integrity (this reads every file) ...")
    statuses, widths, heights, corrupt = check_images(df)
    df["status"] = statuses
    df["width"] = widths
    df["height"] = heights
    ok = df[df["status"] == "ok"]
    print(f"ok={len(ok)} corrupt={len(corrupt)}")

    dup_groups = exact_duplicates(df)
    print(f"exact duplicate groups: {len(dup_groups)}")

    print("Parsing damage labels ...")
    labels_df = parse_damage_labels()
    label_stats = label_statistics(labels_df)
    stats_global.update(label_stats)

    stats = {
        "total_images": int(len(df)),
        "per_split": df["split"].value_counts().to_dict(),
        "per_split_source": df.groupby(["split", "source"]).size().unstack(fill_value=0).to_dict(),
        "image_sizes": {
            f"{w}x{h}": int(n)
            for (w, h), n in df.groupby(["width", "height"]).size().items()
        },
        "corrupt_images": corrupt,
        "exact_duplicate_groups": dup_groups,
        "labels": label_stats,
        "notes": [
            "Damage labels are only available for train/Czech and train/India "
            "(VOC xml annotations). train/Japan labels were lost when the "
            "upstream train.tar.gz download was truncated; test1/test2 ship "
            "without damage annotations.",
            "test1/test2 still carry source-country folder names and are used "
            "only for dataset description, not for damage evaluation.",
        ],
    }

    make_figures(df, labels_df)

    save_json(stats, Config.results_dir / "metrics" / "dataset_stats.json")
    df.drop(columns="path").assign(path=[str(p) for p in df["path"]]).to_csv(
        Config.results_dir / "metrics" / "image_inventory.csv", index=False
    )
    labels_df.drop(columns="path").assign(path=[str(p) for p in labels_df["path"]]).to_csv(
        Config.results_dir / "metrics" / "label_table.csv", index=False
    )
    print("Saved dataset stats + figures to", Config.results_dir)
    return stats


if __name__ == "__main__":
    main()

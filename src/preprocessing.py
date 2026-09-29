"""
RDD2020 Pascal VOC -> 4-class image classification preprocessing.

Selected classes:
    D00 - Longitudinal Crack
    D10 - Transverse Crack
    D20 - Alligator Crack
    D40 - Pothole

Important:
    The train/validation/test split is performed at ORIGINAL IMAGE level
    before bounding-box crops are generated. This prevents crops from the
    same source image appearing in multiple splits.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASSES = ["D00", "D10", "D20", "D40"]

CLASS_NAMES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def parse_annotation(xml_path: Path):
    """Parse one Pascal VOC XML file."""

    root = ET.parse(xml_path).getroot()

    filename_element = root.find("filename")
    filename = filename_element.text.strip() if filename_element is not None else None

    objects = []

    for obj in root.findall("object"):
        name_element = obj.find("name")

        if name_element is None or name_element.text is None:
            continue

        label = name_element.text.strip()

        if label not in CLASSES:
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        try:
            xmin = int(float(bbox.findtext("xmin")))
            ymin = int(float(bbox.findtext("ymin")))
            xmax = int(float(bbox.findtext("xmax")))
            ymax = int(float(bbox.findtext("ymax")))
        except (TypeError, ValueError):
            continue

        objects.append(
            {
                "label": label,
                "bbox": (xmin, ymin, xmax, ymax),
            }
        )

    return filename, objects


def discover_annotations(raw_train_dir: Path):
    """Find all XML/image pairs under country directories."""

    records = []

    xml_files = sorted(raw_train_dir.rglob("annotations/xmls/*.xml"))

    for xml_path in xml_files:

        country_dir = xml_path.parents[2]
        image_dir = country_dir / "images"

        filename, objects = parse_annotation(xml_path)

        if not filename:
            continue

        image_path = image_dir / filename

        if not image_path.exists():
            # Try stem matching if the XML filename does not directly resolve.
            candidates = list(image_dir.glob(f"{Path(filename).stem}.*"))

            if candidates:
                image_path = candidates[0]
            else:
                continue

        selected_objects = [
            obj for obj in objects if obj["label"] in CLASSES
        ]

        records.append(
            {
                "xml_path": xml_path,
                "image_path": image_path,
                "objects": selected_objects,
                "country": country_dir.name,
            }
        )

    return records


def create_split(records, seed=42):
    """
    Randomly split ORIGINAL images.

    Each original image stays entirely within one split.
    """

    rng = random.Random(seed)

    records = list(records)
    rng.shuffle(records)

    total = len(records)

    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }


def clean_output(output_dir: Path):
    """Remove previous generated classification dataset."""

    if output_dir.exists():
        print(f"Removing previous output: {output_dir}")
        shutil.rmtree(output_dir)


def create_directories(output_dir: Path):

    for split in SPLIT_RATIOS:
        for label in CLASSES:
            (output_dir / split / label).mkdir(
                parents=True,
                exist_ok=True,
            )


def clamp_bbox(bbox, width, height):

    xmin, ymin, xmax, ymax = bbox

    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(1, min(xmax, width))
    ymax = max(1, min(ymax, height))

    return xmin, ymin, xmax, ymax


def add_padding(bbox, width, height, padding_ratio=0.10):

    xmin, ymin, xmax, ymax = bbox

    box_width = xmax - xmin
    box_height = ymax - ymin

    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)

    xmin -= pad_x
    ymin -= pad_y
    xmax += pad_x
    ymax += pad_y

    return clamp_bbox(
        (xmin, ymin, xmax, ymax),
        width,
        height,
    )


def generate_crops(
    split_records,
    output_dir,
    manifest_writer,
    padding_ratio=0.10,
):

    crop_counts = Counter()
    source_image_counts = Counter()

    invalid_bbox_count = 0
    failed_image_count = 0

    for split, records in split_records.items():

        for record in records:

            image_path = record["image_path"]

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                print(f"WARNING: Could not open {image_path}: {exc}")
                failed_image_count += 1
                continue

            width, height = image.size

            source_has_crop = False

            for object_index, obj in enumerate(record["objects"]):

                label = obj["label"]

                bbox = add_padding(
                    obj["bbox"],
                    width,
                    height,
                    padding_ratio,
                )

                xmin, ymin, xmax, ymax = bbox

                if xmax <= xmin or ymax <= ymin:
                    invalid_bbox_count += 1
                    continue

                crop = image.crop((xmin, ymin, xmax, ymax))

                crop_filename = (
                    f"{image_path.stem}"
                    f"__obj{object_index:03d}"
                    f"__{label}.jpg"
                )

                crop_path = (
                    output_dir
                    / split
                    / label
                    / crop_filename
                )

                crop.save(
                    crop_path,
                    format="JPEG",
                    quality=95,
                )

                crop_counts[(split, label)] += 1
                source_has_crop = True

                manifest_writer.writerow(
                    {
                        "split": split,
                        "class": label,
                        "crop_path": str(crop_path),
                        "source_image": str(image_path),
                        "source_xml": str(record["xml_path"]),
                        "country": record["country"],
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                )

            if source_has_crop:
                source_image_counts[split] += 1

    return crop_counts, source_image_counts, invalid_bbox_count, failed_image_count


def print_report(
    all_records,
    split_records,
    crop_counts,
    source_image_counts,
    invalid_bbox_count,
    failed_image_count,
):

    print("\n" + "=" * 70)
    print("RDD2020 PREPROCESSING VERIFICATION")
    print("=" * 70)

    print(f"\nOriginal XML/image records found: {len(all_records)}")

    selected_records = [
        record
        for record in all_records
        if len(record["objects"]) > 0
    ]

    print(
        f"Original images containing at least one selected class: "
        f"{len(selected_records)}"
    )

    print("\nOriginal-image split:")
    for split, records in split_records.items():
        print(f"  {split:5s}: {len(records):6d} images")

    print("\nGenerated crops:")

    total_crops = 0

    for split in ["train", "val", "test"]:

        print(f"\n  {split.upper()}")

        split_total = 0

        for label in CLASSES:
            count = crop_counts[(split, label)]
            split_total += count
            total_crops += count

            print(
                f"    {label}: "
                f"{count:6d} crops"
            )

        print(f"    TOTAL: {split_total:6d} crops")

    print("\nOverall crop counts:")

    for label in CLASSES:
        total = sum(
            crop_counts[(split, label)]
            for split in ["train", "val", "test"]
        )

        print(
            f"  {label}: "
            f"{total:6d} crops"
        )

    print(f"  {'TOTAL':5s}: {total_crops:6d} crops")

    print("\nSource images containing crops:")

    for split in ["train", "val", "test"]:
        print(
            f"  {split:5s}: "
            f"{source_image_counts[split]:6d}"
        )

    print("\nValidation information:")
    print(f"  Invalid bounding boxes: {invalid_bbox_count}")
    print(f"  Failed images:          {failed_image_count}")

    print("\nExpected raw selected-object counts:")
    print("  D00: 6592")
    print("  D10: 4446")
    print("  D20: 8381")
    print("  D40: 5627")
    print("  TOTAL: 25046")

    expected = {
        "D00": 6592,
        "D10": 4446,
        "D20": 8381,
        "D40": 5627,
    }

    actual = {
        label: sum(
            crop_counts[(split, label)]
            for split in ["train", "val", "test"]
        )
        for label in CLASSES
    }

    print("\nCount verification:")

    all_match = True

    for label in CLASSES:

        if actual[label] == expected[label]:
            status = "PASS"
        else:
            status = "CHECK"

        if actual[label] != expected[label]:
            all_match = False

        print(
            f"  {label}: "
            f"expected={expected[label]:6d}, "
            f"actual={actual[label]:6d} "
            f"[{status}]"
        )

    print(
        f"\nOverall verification: "
        f"{'PASS' if all_match else 'CHECK COUNTS'}"
    )

    print("=" * 70)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-train",
        type=Path,
        required=True,
        help="Path to extracted RDD2020 train directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/classification"),
        help="Output classification dataset directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Bounding-box padding ratio",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output before preprocessing",
    )

    args = parser.parse_args()

    print("Raw training directory:")
    print(args.raw_train)

    print("\nOutput directory:")
    print(args.output)

    if not args.raw_train.exists():
        raise FileNotFoundError(
            f"Raw training directory does not exist: {args.raw_train}"
        )

    if args.clean:
        clean_output(args.output)

    create_directories(args.output)

    print("\nDiscovering annotations...")

    records = discover_annotations(args.raw_train)

    print(f"Found {len(records)} XML/image records.")

    split_records = create_split(
        records,
        seed=args.seed,
    )

    manifest_path = args.output / "manifest.csv"

    print(f"\nWriting manifest:")
    print(manifest_path)

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest_file:

        fieldnames = [
            "split",
            "class",
            "crop_path",
            "source_image",
            "source_xml",
            "country",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        ]

        writer = csv.DictWriter(
            manifest_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        (
            crop_counts,
            source_image_counts,
            invalid_bbox_count,
            failed_image_count,
        ) = generate_crops(
            split_records,
            args.output,
            writer,
            padding_ratio=args.padding,
        )

    print_report(
        records,
        split_records,
        crop_counts,
        source_image_counts,
        invalid_bbox_count,
        failed_image_count,
    )


if __name__ == "__main__":
    main()
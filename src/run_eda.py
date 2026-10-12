"""Run a reproducible audit of the raw COCO LED dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/led_lights_coco"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/eda_report.json"))
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(root: Path, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    annotation_path = root / split / "_annotations.coco.json"
    with annotation_path.open(encoding="utf-8") as file:
        coco = json.load(file)
    images = {image["id"]: image for image in coco["images"]}
    return coco, images


def summarize_split(root: Path, split: str) -> dict[str, Any]:
    coco, images = load_split(root, split)
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    box_sizes: list[dict[str, float]] = []
    color_samples: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for annotation in coco["annotations"]:
        category_name = categories[annotation["category_id"]]
        class_counts[category_name] += 1
        annotations_by_image[annotation["image_id"]].append(annotation)
        _, _, width, height = annotation["bbox"]
        box_sizes.append({"width": width, "height": height, "area": width * height})

        image = images[annotation["image_id"]]
        image_path = root / split / image["file_name"]
        with Image.open(image_path).convert("RGB") as opened_image:
            x, y, _, _ = annotation["bbox"]
            crop = opened_image.crop((x, y, x + width, y + height))
            color_samples[category_name].append(tuple(ImageStat.Stat(crop).mean))

    image_records = []
    for image_id, image in images.items():
        image_path = root / split / image["file_name"]
        with Image.open(image_path) as opened_image:
            width, height = opened_image.size
            mode = opened_image.mode
        image_records.append(
            {
                "file_name": image["file_name"],
                "width": width,
                "height": height,
                "mode": mode,
                "annotation_count": len(annotations_by_image[image_id]),
                "labels": sorted(
                    categories[annotation["category_id"]]
                    for annotation in annotations_by_image[image_id]
                ),
                "sha256": file_hash(image_path),
            }
        )

    return {
        "image_count": len(image_records),
        "annotation_count": len(coco["annotations"]),
        "categories": categories,
        "class_counts": dict(sorted(class_counts.items())),
        "images_without_annotations": sum(
            record["annotation_count"] == 0 for record in image_records
        ),
        "image_dimensions": {
            f"{width}x{height}": count
            for (width, height), count in Counter(
                (record["width"], record["height"]) for record in image_records
            ).items()
        },
        "color_modes": dict(sorted(Counter(record["mode"] for record in image_records).items())),
        "box_size_summary": {
            "count": len(box_sizes),
            "min_width": min((box["width"] for box in box_sizes), default=0),
            "max_width": max((box["width"] for box in box_sizes), default=0),
            "min_height": min((box["height"] for box in box_sizes), default=0),
            "max_height": max((box["height"] for box in box_sizes), default=0),
            "min_area": min((box["area"] for box in box_sizes), default=0),
            "max_area": max((box["area"] for box in box_sizes), default=0),
        },
        "bbox_mean_rgb_by_label": {
            label: {
                "count": len(samples),
                "mean_r": sum(sample[0] for sample in samples) / len(samples),
                "mean_g": sum(sample[1] for sample in samples) / len(samples),
                "mean_b": sum(sample[2] for sample in samples) / len(samples),
            }
            for label, samples in sorted(color_samples.items())
        },
        "images": sorted(image_records, key=lambda record: record["file_name"]),
    }


def duplicate_summary(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hashes: dict[str, list[str]] = defaultdict(list)
    stems: dict[str, list[str]] = defaultdict(list)
    for split, summary in splits.items():
        for image in summary["images"]:
            hashes[image["sha256"]].append(f"{split}/{image['file_name']}")
            source_stem = image["file_name"].split("_jpg.rf.")[0].split("_png.rf.")[0]
            stems[source_stem].append(f"{split}/{image['file_name']}")

    return {
        "exact_duplicate_groups": sorted(
            [paths for paths in hashes.values() if len(paths) > 1], key=lambda paths: paths[0]
        ),
        "source_stem_groups": sorted(
            [paths for paths in stems.values() if len(paths) > 1], key=lambda paths: paths[0]
        ),
    }


def main() -> None:
    args = parse_args()
    splits = {split: summarize_split(args.raw_root, split) for split in ("train", "valid")}
    train_names = {image["file_name"] for image in splits["train"]["images"]}
    valid_names = {image["file_name"] for image in splits["valid"]["images"]}
    report = {
        "raw_root": str(args.raw_root),
        "splits": splits,
        "duplicates": duplicate_summary(splits),
        "cross_split_filename_overlap": sorted(train_names & valid_names),
        "binary_label_mapping": {"green": 0, "yellow": 1},
        "excluded_labels": ["blue"],
        "limitations": [
            "COCO object annotations do not establish that each image has one independently sampled device.",
            "Filename and exact-pixel duplicate checks cannot prove absence of near-duplicate views or session leakage.",
            "LED color is a proxy for server status, not proof of server health.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "splits": splits, "duplicates": report["duplicates"]}, indent=2))


if __name__ == "__main__":
    main()
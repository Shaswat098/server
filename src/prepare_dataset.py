"""Create padded, resized binary LED crops with an image-grouped split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class SourceImage:
    source_split: str
    image_id: int
    file_name: str
    annotations: tuple[dict[str, Any], ...]

    @property
    def labels(self) -> set[str]:
        return {annotation["label"] for annotation in self.annotations}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/led_lights_coco"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/led_crops"))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--padding-fraction", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=64)
    return parser.parse_args()


def load_source_images(raw_root: Path) -> list[SourceImage]:
    source_images: list[SourceImage] = []
    for source_split in ("train", "valid"):
        split_root = raw_root / source_split
        coco = json.loads((split_root / "_annotations.coco.json").read_text(encoding="utf-8"))
        categories = {category["id"]: category["name"] for category in coco["categories"]}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[annotation["image_id"]].append(
                {**annotation, "label": categories[annotation["category_id"]]}
            )
        for image in coco["images"]:
            source_images.append(
                SourceImage(
                    source_split=source_split,
                    image_id=image["id"],
                    file_name=image["file_name"],
                    annotations=tuple(annotations_by_image[image["id"]]),
                )
            )
    return source_images


def assign_splits(images: list[SourceImage], seed: int) -> dict[str, list[SourceImage]]:
    eligible = [image for image in images if image.labels & {"green", "yellow"}]
    yellow_groups = [image for image in eligible if "yellow" in image.labels]
    other_groups = [image for image in eligible if "yellow" not in image.labels]
    randomizer = random.Random(seed)
    randomizer.shuffle(yellow_groups)
    randomizer.shuffle(other_groups)

    assigned: dict[str, list[SourceImage]] = {"train": [], "valid": [], "test": []}
    for index, image in enumerate(yellow_groups):
        split = ("valid", "test", "train")[index] if index < 3 else "train"
        assigned[split].append(image)

    target_ratios = {"train": 0.70, "valid": 0.15, "test": 0.15}
    target_counts = {
        split: max(1, round(len(eligible) * ratio)) for split, ratio in target_ratios.items()
    }
    for image in other_groups:
        split = min(
            assigned,
            key=lambda candidate: len(assigned[candidate]) / target_counts[candidate],
        )
        assigned[split].append(image)
    return assigned


def crop_image(
    image: Image.Image,
    bbox: list[float],
    padding_fraction: float,
    size: int,
) -> Image.Image:
    x, y, width, height = bbox
    padding = padding_fraction * max(width, height)
    left = max(0, int(x - padding))
    top = max(0, int(y - padding))
    right = min(image.width, int(x + width + padding + 0.999))
    bottom = min(image.height, int(y + height + padding + 0.999))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop bounds for bbox {bbox}")
    return image.crop((left, top, right, bottom)).resize((size, size), Image.Resampling.LANCZOS)


def process_split(
    split: str,
    images: list[SourceImage],
    raw_root: Path,
    output_root: Path,
    padding_fraction: float,
    size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_image in sorted(images, key=lambda item: (item.source_split, item.file_name)):
        source_path = raw_root / source_image.source_split / source_image.file_name
        with Image.open(source_path).convert("RGB") as opened_image:
            for annotation in source_image.annotations:
                label = annotation["label"]
                if label not in {"green", "yellow"}:
                    continue
                crop = crop_image(opened_image, annotation["bbox"], padding_fraction, size)
                digest = hashlib.sha256(
                    f"{source_image.source_split}/{source_image.file_name}/{annotation['id']}".encode()
                ).hexdigest()[:12]
                output_name = f"{label}_{digest}.jpg"
                output_path = output_root / split / label / output_name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(output_path, format="JPEG", quality=95)
                records.append(
                    {
                        "path": str(output_path),
                        "split": split,
                        "label": label,
                        "class_index": {"green": 0, "yellow": 1}[label],
                        "source_split": source_image.source_split,
                        "source_file": source_image.file_name,
                        "source_image_id": source_image.image_id,
                        "annotation_id": annotation["id"],
                        "bbox": annotation["bbox"],
                        "padding_fraction": padding_fraction,
                        "size": size,
                    }
                )
    return records


def main() -> None:
    args = parse_args()
    if not 0 <= args.padding_fraction <= 1:
        raise ValueError("padding fraction must be between 0 and 1")
    if args.size < 1:
        raise ValueError("size must be positive")
    source_images = load_source_images(args.raw_root)
    assigned = assign_splits(source_images, args.seed)
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    all_records: list[dict[str, Any]] = []
    for split, split_images in assigned.items():
        all_records.extend(
            process_split(
                split,
                split_images,
                args.raw_root,
                args.output_root,
                args.padding_fraction,
                args.size,
            )
        )

    manifest = {
        "seed": args.seed,
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "crop_policy": {"padding_fraction": args.padding_fraction, "size": args.size},
        "binary_label_mapping": {"green": 0, "yellow": 1},
        "excluded_labels": ["blue"],
        "source_image_counts": {split: len(images) for split, images in assigned.items()},
        "source_images_excluded_blue_only": sum(
            not (image.labels & {"green", "yellow"}) for image in source_images
        ),
        "records": all_records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_root": str(args.output_root),
        "source_image_counts": manifest["source_image_counts"],
        "crop_counts": dict(Counter(record["split"] for record in all_records)),
        "label_counts": dict(Counter(record["label"] for record in all_records)),
        "blue_only_images_excluded": manifest["source_images_excluded_blue_only"],
    }, indent=2))


if __name__ == "__main__":
    main()
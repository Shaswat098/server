"""Measure brightness, color concentration, and fixed-crop coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32)
    brightness = image.mean(axis=2)
    peak_threshold = max(32.0, float(np.percentile(brightness, 99.5)))
    peak_mask = brightness >= peak_threshold
    rows, columns = np.where(peak_mask)
    if rows.size:
        peak_bounds = {
            "left": int(columns.min()),
            "top": int(rows.min()),
            "right": int(columns.max()),
            "bottom": int(rows.max()),
        }
    else:
        peak_bounds = None

    crop_mask = np.zeros(brightness.shape, dtype=bool)
    crop_mask[:, 20:140] = True
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    color_scores = {
        "red": float(np.mean(red - (green + blue) / 2)),
        "green": float(np.mean(green - (red + blue) / 2)),
        "blue": float(np.mean(blue - (red + green) / 2)),
    }
    report = {
        "image": str(args.image),
        "shape": list(image.shape),
        "brightness": {
            "mean": float(brightness.mean()),
            "maximum": float(brightness.max()),
            "p99_5": float(np.percentile(brightness, 99.5)),
            "peak_threshold": peak_threshold,
            "peak_pixel_count": int(peak_mask.sum()),
        },
        "brightest_region": peak_bounds,
        "fixed_crop": {
            "left": 20,
            "right_exclusive": 140,
            "peak_pixels_inside": int((peak_mask & crop_mask).sum()),
            "peak_coverage": (
                float((peak_mask & crop_mask).sum() / peak_mask.sum())
                if peak_mask.any()
                else 0.0
            ),
        },
        "mean_color_dominance": color_scores,
        "limitations": [
            "Brightness and color concentration do not identify the LED semantic label.",
            "A known-color, adequately exposed capture is required for crop-policy decisions.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

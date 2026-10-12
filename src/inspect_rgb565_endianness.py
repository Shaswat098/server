"""Decode a raw RGB565 frame using both byte orders and compare exposure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


WIDTH = 160
HEIGHT = 120


def decode(raw: bytes, byte_order: str) -> np.ndarray:
    packed = np.frombuffer(raw, dtype=byte_order + "u2").reshape(HEIGHT, WIDTH)
    red = ((packed & 0xF800) >> 11).astype(np.uint8)
    green = ((packed & 0x07E0) >> 5).astype(np.uint8)
    blue = (packed & 0x001F).astype(np.uint8)
    return np.stack(
        (
            red * 255 // 31,
            green * 255 // 63,
            blue * 255 // 31,
        ),
        axis=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.read_bytes()
    expected_bytes = WIDTH * HEIGHT * 2
    if len(raw) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes, received {len(raw)}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, byte_order in (("big_endian", ">"), ("little_endian", "<")):
        image = decode(raw, byte_order)
        Image.fromarray(image, mode="RGB").save(args.output_dir / f"{name}.png")
        brightness = image.mean(axis=2)
        print(
            f"{name}: mean={brightness.mean():.2f}, "
            f"maximum={brightness.max():.2f}, "
            f"p99.5={np.percentile(brightness, 99.5):.2f}"
        )


if __name__ == "__main__":
    main()

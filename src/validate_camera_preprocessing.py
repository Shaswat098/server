"""Validate the deterministic RGB565-to-INT8 camera preprocessing contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FRAME_WIDTH = 160
FRAME_HEIGHT = 120
CROP_SIZE = 120
CROP_LEFT = 20
MODEL_SIZE = 64
INPUT_SCALE = 0.003921568859368563
INPUT_ZERO_POINT = -128


def pack_rgb565(rgb: np.ndarray) -> np.ndarray:
    red = np.rint(rgb[..., 0] * 31 / 255).astype(np.uint16)
    green = np.rint(rgb[..., 1] * 63 / 255).astype(np.uint16)
    blue = np.rint(rgb[..., 2] * 31 / 255).astype(np.uint16)
    return (red << 11) | (green << 5) | blue


def preprocess_embedded_equivalent(frame_rgb565: np.ndarray) -> np.ndarray:
    output = np.empty((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.int8)
    for y in range(MODEL_SIZE):
        source_y = y * CROP_SIZE // MODEL_SIZE
        for x in range(MODEL_SIZE):
            source_x = CROP_LEFT + x * CROP_SIZE // MODEL_SIZE
            pixel = int(frame_rgb565[source_y, source_x])
            channels = (
                (pixel & 0xF800) >> 11,
                (pixel & 0x07E0) >> 5,
                pixel & 0x001F,
            )
            scales = (31.0, 63.0, 31.0)
            for channel, (value, channel_scale) in enumerate(
                zip(channels, scales)
            ):
                normalized = (value / channel_scale) / INPUT_SCALE
                output[y, x, channel] = np.clip(
                    np.rint(normalized + INPUT_ZERO_POINT), -128, 127
                )
    return output


def main() -> None:
    rows, columns = np.indices((FRAME_HEIGHT, FRAME_WIDTH))
    rgb = np.stack(
        (
            (columns * 3 + rows) % 256,
            (rows * 5 + columns * 2) % 256,
            (rows * 7 + columns * 11) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    packed = pack_rgb565(rgb)
    processed = preprocess_embedded_equivalent(packed)

    expected_source_coordinates = {
        "top_left": [0, CROP_LEFT],
        "bottom_right": [
            (MODEL_SIZE - 1) * CROP_SIZE // MODEL_SIZE,
            CROP_LEFT + (MODEL_SIZE - 1) * CROP_SIZE // MODEL_SIZE,
        ],
    }
    report = {
        "frame_shape": [FRAME_HEIGHT, FRAME_WIDTH],
        "crop": {
            "left": CROP_LEFT,
            "top": 0,
            "width": CROP_SIZE,
            "height": CROP_SIZE,
        },
        "resize": "nearest_neighbor_with_source_index=floor(output*120/64)",
        "input_quantization": {
            "scale": INPUT_SCALE,
            "zero_point": INPUT_ZERO_POINT,
            "dtype": "int8",
        },
        "expected_source_coordinates": expected_source_coordinates,
        "output_shape": list(processed.shape),
        "output_range": [int(processed.min()), int(processed.max())],
        "checks": {
            "crop_in_bounds": bool(
                CROP_LEFT >= 0 and CROP_LEFT + CROP_SIZE <= FRAME_WIDTH
            ),
            "output_dtype_int8": str(processed.dtype) == "int8",
            "output_shape_64x64x3": processed.shape == (64, 64, 3),
            "quantized_range_valid": bool(
                processed.min() >= -128 and processed.max() <= 127
            ),
        },
        "limitations": [
            "This is a deterministic contract test, not a validation of LED location.",
            "It does not prove equivalence to the annotated crop and Lanczos training pipeline.",
            "A real camera capture fixture is still required before selecting the crop policy.",
        ],
    }
    if not all(report["checks"].values()):
        raise RuntimeError(json.dumps(report, indent=2))

    output_path = Path("data/interim/camera_preprocessing_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

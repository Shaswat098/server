"""Create TFLite Micro model and parity fixture headers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def format_bytes(values: bytes, values_per_line: int = 12) -> str:
    lines = []
    for offset in range(0, len(values), values_per_line):
        chunk = values[offset : offset + values_per_line]
        lines.append("    " + ", ".join(f"0x{value:02x}" for value in chunk))
    return ",\n".join(lines)


def format_float_array(values: np.ndarray, values_per_line: int = 6) -> str:
    flattened = values.astype(np.float32).reshape(-1)
    lines = []
    for offset in range(0, len(flattened), values_per_line):
        chunk = flattened[offset : offset + values_per_line]
        lines.append("    " + ", ".join(f"{value:.9g}f" for value in chunk))
    return ",\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("data/interim/tflite/candidate_b_float32.tflite"))
    parser.add_argument("--inputs", type=Path, default=Path("data/interim/tflite/candidate_b_export_inputs.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("embedded"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_bytes = args.model.read_bytes()
    arrays = dict(np.load(args.inputs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_header = """#pragma once
#include <cstddef>
#include <cstdint>

alignas(16) const unsigned char kCandidateBModel[] = {{
{model_bytes}
}};
const size_t kCandidateBModelSize = sizeof(kCandidateBModel);
""".format(model_bytes=format_bytes(model_bytes))
    (args.output_dir / "model_data.h").write_text(model_header, encoding="ascii")
    fixture_header = """#pragma once
#include <cstddef>

constexpr size_t kParitySampleCount = {sample_count};
constexpr size_t kParityInputElements = 64 * 64 * 3;
constexpr size_t kParityOutputElements = 2;
const float kParityInputs[] = {{
{inputs}
}};
const float kParityExpectedLogits[] = {{
{logits}
}};
""".format(
        sample_count=arrays["test_inputs"].shape[0],
        inputs=format_float_array(arrays["test_inputs"].transpose(0, 2, 3, 1)),
        logits=format_float_array(arrays["reference_logits"]),
    )
    (args.output_dir / "parity_fixture.h").write_text(fixture_header, encoding="ascii")
    metadata = {
        "model_header": str(args.output_dir / "model_data.h"),
        "model_bytes": len(model_bytes),
        "input_contract": {"shape": [64, 64, 3], "dtype": "float32", "range": [0.0, 1.0]},
        "parity_fixture": str(args.output_dir / "parity_fixture.h"),
        "parity_samples": int(arrays["test_inputs"].shape[0]),
    }
    (args.output_dir / "bundle_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
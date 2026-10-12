"""Export Candidate B as full-integer TFLite using train-only calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from export_tflite import build_model, run_tflite


DEFAULT_WEIGHTS = Path("data/interim/tflite/candidate_b_export_inputs.npz")
DEFAULT_MANIFEST = Path("data/processed/led_crops/manifest.json")
DEFAULT_OUTPUT = Path("data/interim/tflite/candidate_b_int8.tflite")
DEFAULT_EMBEDDED_OUTPUT = Path("embedded/int8_bundle")
INPUT_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedded-output", type=Path, default=DEFAULT_EMBEDDED_OUTPUT)
    return parser.parse_args()


def load_record(record: dict[str, Any]) -> np.ndarray:
    with Image.open(record["path"]) as opened_image:
        image = opened_image.convert("RGB").resize(
            (INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def metric_report(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(logits, axis=1)
    matrix = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    yellow_true_positive = int(matrix[1, 1])
    yellow_predicted = int(matrix[:, 1].sum())
    yellow_actual = int(matrix[1, :].sum())
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "precision_yellow": (
            yellow_true_positive / yellow_predicted if yellow_predicted else 0.0
        ),
        "recall_yellow": (
            yellow_true_positive / yellow_actual if yellow_actual else 0.0
        ),
        "confusion_matrix": matrix.astype(int).tolist(),
        "sample_count": int(labels.size),
        "yellow_count": int(yellow_actual),
    }


def write_int8_fixture(
    output_dir: Path,
    inputs: np.ndarray,
    logits: np.ndarray,
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
) -> None:
    quantized_inputs = np.clip(
        np.round(inputs / input_scale + input_zero_point), -128, 127
    ).astype(np.int8)
    quantized_outputs = np.clip(
        np.round(logits / output_scale + output_zero_point), -128, 127
    ).astype(np.int8)

    def format_values(values: np.ndarray) -> str:
        flattened = values.reshape(-1)
        lines = []
        for offset in range(0, len(flattened), 16):
            lines.append(
                "    " + ", ".join(f"{int(value)}" for value in flattened[offset : offset + 16])
            )
        return ",\n".join(lines)

    header = f"""#pragma once
#include <cstddef>
#include <cstdint>

constexpr size_t kInt8ParitySampleCount = {inputs.shape[0]};
constexpr size_t kInt8ParityInputElements = 64 * 64 * 3;
constexpr size_t kInt8ParityOutputElements = 2;
const int8_t kInt8ParityInputs[] = {{
{format_values(quantized_inputs)}
}};
const int8_t kInt8ParityExpectedOutputs[] = {{
{format_values(quantized_outputs)}
}};
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parity_fixture.h").write_text(header, encoding="ascii")


def main() -> None:
    args = parse_args()
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow is required for INT8 TFLite export.") from error

    arrays = dict(np.load(args.weights))
    model = build_model(tf, arrays)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_records = [record for record in manifest["records"] if record["split"] == "train"]
    test_records = [record for record in manifest["records"] if record["split"] == "test"]
    train_inputs = np.stack([load_record(record) for record in train_records])
    test_inputs = np.stack([load_record(record) for record in test_records])
    test_labels = np.asarray([record["class_index"] for record in test_records], dtype=np.int64)

    def representative_dataset() -> Any:
        for sample in train_inputs:
            yield [sample[None, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tflite_bytes)

    interpreter = tf.lite.Interpreter(model_path=str(args.output))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if input_details["dtype"] is not np.int8 or output_details["dtype"] is not np.int8:
        raise RuntimeError(
            f"Expected int8 input/output, got {input_details['dtype']} and "
            f"{output_details['dtype']}."
        )
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]
    int8_outputs = []
    for sample in test_inputs:
        quantized = np.round(sample / input_scale + input_zero_point)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quantized[None, ...])
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"])[0]
        int8_outputs.append(
            (raw_output.astype(np.float32) - output_zero_point) * output_scale
        )
    int8_logits = np.stack(int8_outputs)
    float_logits = run_tflite(
        tf, Path("data/interim/tflite/candidate_b_float32.tflite"), test_inputs
    )
    report = {
        "model": "Candidate B depthwise-separable CNN",
        "format": "full integer TFLite",
        "calibration_policy": "train split only",
        "calibration_samples": len(train_inputs),
        "test_samples": len(test_inputs),
        "input": {
            "shape": [int(value) for value in input_details["shape"]],
            "dtype": str(input_details["dtype"]),
            "scale": float(input_scale),
            "zero_point": int(input_zero_point),
        },
        "output": {
            "shape": [int(value) for value in output_details["shape"]],
            "dtype": str(output_details["dtype"]),
            "scale": float(output_scale),
            "zero_point": int(output_zero_point),
        },
        "tflite_size_bytes": len(tflite_bytes),
        "test_metrics": metric_report(int8_logits, test_labels),
        "max_abs_logit_delta_vs_float32_tflite": float(
            np.max(np.abs(int8_logits - float_logits))
        ),
        "limitations": [
            "The test set contains only two yellow crops.",
            "LED color is a proxy for server status, not proof of server health.",
        ],
    }
    write_int8_fixture(
        args.embedded_output,
        test_inputs,
        int8_logits,
        float(input_scale),
        int(input_zero_point),
        float(output_scale),
        int(output_zero_point),
    )
    report_path = args.output.with_name("int8_desktop_verification.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

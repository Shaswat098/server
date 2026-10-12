"""Export Candidate B to TFLite and verify desktop outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("data/interim/tflite/candidate_b_export_inputs.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/tflite"))
    return parser.parse_args()


def build_model(tf: object, weights: dict[str, np.ndarray]) -> object:
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(64, 64, 3), dtype=tf.float32, name="input")
    tensor = inputs
    for index, (input_channels, output_channels) in enumerate(((3, 8), (8, 16), (16, 32))):
        tensor = layers.DepthwiseConv2D(3, padding="same", use_bias=True, name=f"depthwise_{index}")(tensor)
        tensor = layers.ReLU()(tensor)
        tensor = layers.Conv2D(output_channels, 1, padding="valid", use_bias=True, name=f"pointwise_{index}")(tensor)
        tensor = layers.ReLU()(tensor)
        if index < 2:
            tensor = layers.MaxPool2D(2)(tensor)
    tensor = layers.GlobalAveragePooling2D()(tensor)
    outputs = layers.Dense(2, name="classifier")(tensor)
    model = tf.keras.Model(inputs, outputs)
    for index in range(3):
        depthwise = weights[f"features__{index * 2}__block__0__weight"].transpose(2, 3, 0, 1)
        depthwise_bias = weights[f"features__{index * 2}__block__0__bias"]
        pointwise = weights[f"features__{index * 2}__block__2__weight"].transpose(2, 3, 1, 0)
        pointwise_bias = weights[f"features__{index * 2}__block__2__bias"]
        model.get_layer(f"depthwise_{index}").set_weights([depthwise, depthwise_bias])
        model.get_layer(f"pointwise_{index}").set_weights([pointwise, pointwise_bias])
    model.get_layer("classifier").set_weights([
        weights["classifier__weight"].transpose(1, 0),
        weights["classifier__bias"],
    ])
    return model


def run_tflite(tf: object, path: Path, inputs: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    outputs = []
    for sample in inputs:
        interpreter.set_tensor(input_details["index"], sample[None, ...].astype(np.float32))
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_details["index"])[0])
    return np.stack(outputs)


def main() -> None:
    args = parse_args()
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for TFLite export. Install it in the workspace .venv "
            "before rerunning this script."
        ) from error

    arrays = dict(np.load(args.weights))
    model = build_model(tf, arrays)
    inputs = arrays["test_inputs"].transpose(0, 2, 3, 1)
    reference_logits = arrays["reference_logits"]
    keras_logits = model(inputs, training=False).numpy()
    keras_max_error = float(np.max(np.abs(keras_logits - reference_logits)))
    if keras_max_error > 1e-4:
        raise RuntimeError(f"Keras/PyTorch mapping error is {keras_max_error}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = args.output_dir / "candidate_b_float32.tflite"
    tflite_path.write_bytes(tflite_bytes)
    tflite_logits = run_tflite(tf, tflite_path, inputs)
    tflite_max_error = float(np.max(np.abs(tflite_logits - reference_logits)))
    if tflite_max_error > 1e-4:
        raise RuntimeError(f"TFLite/PyTorch parity error is {tflite_max_error}")
    report = {
        "model": "Candidate B depthwise-separable CNN",
        "input": {"shape": [64, 64, 3], "dtype": "float32", "range": [0.0, 1.0]},
        "test_sample_count": int(inputs.shape[0]),
        "keras_pytorch_max_abs_logit_error": keras_max_error,
        "tflite_pytorch_max_abs_logit_error": tflite_max_error,
        "tflite_size_bytes": len(tflite_bytes),
        "tflite_path": str(tflite_path),
        "limitations": [
            "This is a float32 TFLite export; embedded INT8 conversion remains a later deployment step.",
            "The test set contains only two yellow crops, so parity does not establish health-classifier quality.",
            "LED color is a proxy for server status, not proof of server health.",
        ],
    }
    (args.output_dir / "desktop_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
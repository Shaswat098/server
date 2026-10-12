"""Classify a close-up server LED image with the verified float32 TFLite model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SIZE = 64
CLASS_NAMES = ("green/healthy", "yellow/fault")
DEFAULT_MODEL = Path("data/interim/tflite/candidate_b_float32.tflite")
BLUE_PIXEL_THRESHOLD = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to a close-up LED image. Omit this option to open the upload window.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a file-picker GUI instead of printing a CLI result.",
    )
    return parser.parse_args()


def load_input(image_path: Path) -> np.ndarray:
    """Match the model contract: RGB, 64x64, float32 values in [0, 1]."""
    with Image.open(image_path) as opened_image:
        rgb_image = opened_image.convert("RGB")
        resized = rgb_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return array[None, ...]


def contains_blue(array: np.ndarray) -> bool:
    """Apply the field policy: a sufficiently visible blue signal is unhealthy."""
    rgb = array[0]
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue_pixels = (
        (blue > 80 / 255)
        & (blue > red * 1.2)
        & (blue > green * 1.1)
        & (blue - red > 20 / 255)
    )
    return float(np.mean(blue_pixels)) >= BLUE_PIXEL_THRESHOLD


def run_inference(model_path: Path, image_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file was not found: {image_path}")

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required. Run this command from the project .venv."
        ) from error

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_array = load_input(image_path)

    expected_shape = tuple(input_details["shape"])
    if expected_shape != input_array.shape:
        raise RuntimeError(
            f"Model input shape {expected_shape} does not match {input_array.shape}."
        )
    if input_details["dtype"] != np.float32:
        raise RuntimeError(
            f"This tool expects the verified float32 model, got {input_details['dtype']}."
        )

    interpreter.set_tensor(input_details["index"], input_array)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details["index"])[0].astype(float)
    blue_detected = contains_blue(input_array)
    model_index = int(np.argmax(logits))
    predicted_index = 1 if blue_detected else model_index
    return {
        "image": str(image_path),
        "prediction": CLASS_NAMES[predicted_index],
        "class_index": predicted_index,
        "model_prediction": CLASS_NAMES[model_index],
        "blue_detected": blue_detected,
        "logits": {
            "green/healthy": round(logits[0], 6),
            "yellow/fault": round(logits[1], 6),
        },
        "warning": (
            "Blue is handled by policy as yellow/fault, not by the trained model. "
            "LED color is a proxy for server status, not proof of server health."
        ),
    }


def run_gui(model_path: Path) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Server LED status classifier")
    root.geometry("560x620")

    selected_image = tk.StringVar(value="No image selected")
    result_text = tk.StringVar(value="Choose a close-up LED image to begin.")
    preview_label = tk.Label(root)
    preview_label.pack(pady=12)
    tk.Label(
        root,
        text="Upload a close-up LED crop (not a full server scene).",
        wraplength=520,
    ).pack()
    tk.Label(root, textvariable=selected_image, wraplength=520).pack(pady=6)
    tk.Label(
        root,
        textvariable=result_text,
        font=("Segoe UI", 14, "bold"),
        wraplength=520,
    ).pack(pady=12)

    def choose_image() -> None:
        image_name = filedialog.askopenfilename(
            title="Choose an LED image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not image_name:
            return
        image_path = Path(image_name)
        selected_image.set(str(image_path))
        try:
            preview = Image.open(image_path).convert("RGB")
            preview.thumbnail((420, 360), Image.Resampling.LANCZOS)
            preview_photo = ImageTk.PhotoImage(preview)
            preview_label.configure(image=preview_photo)
            preview_label.image = preview_photo
            prediction = run_inference(model_path, image_path)
            result_text.set(
                f"Prediction: {prediction['prediction']}\n"
                f"Blue detected: {prediction['blue_detected']}\n"
                f"Logits: {prediction['logits']}"
            )
        except (OSError, RuntimeError, ValueError) as error:
            result_text.set("Classification failed.")
            messagebox.showerror("Classification error", str(error))

    # Import only when the GUI is requested; the CLI remains usable headlessly.
    from PIL import ImageTk

    tk.Button(root, text="Upload image", command=choose_image, width=20).pack(pady=8)
    tk.Label(
        root,
        text="Important: LED color is a proxy for server status, not proof of server health. "
        "Blue is treated as yellow/fault by policy.",
        wraplength=520,
        justify="left",
    ).pack(pady=12)
    root.mainloop()


def main() -> None:
    args = parse_args()
    if args.gui or args.image is None:
        run_gui(args.model)
        return
    result = run_inference(args.model, args.image)
    print(f"Prediction: {result['prediction']}")
    print(f"Blue detected: {result['blue_detected']}")
    print(f"Logits: {result['logits']}")
    print(f"Warning: {result['warning']}", file=sys.stderr)


if __name__ == "__main__":
    main()

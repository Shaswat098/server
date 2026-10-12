"""Compare float32, dynamic-range, post-training INT8, and QAT models."""

from __future__ import annotations

import argparse
import copy
import io
import json
import random
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    get_default_qat_qconfig,
    get_default_qconfig,
    prepare,
    prepare_qat,
)
from torch.utils.data import DataLoader

from compare_architectures import DepthwiseSeparableCNN
from train_baseline import CropDataset, metrics, run_epoch, seed_everything


class QuantizableCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.model = DepthwiseSeparableCNN()
        self.dequant = DeQuantStub()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dequant(self.model(self.quant(inputs)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/led_crops/manifest.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/interim/architecture_comparison/candidate_b_depthwise.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/quantization"))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    return parser.parse_args()


def loaders_for(records: list[dict[str, Any]], seed: int, batch_size: int) -> dict[str, DataLoader]:
    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "valid", "test")
    }
    return {
        split: DataLoader(
            CropDataset(split_records[split]),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        for split in ("train", "valid", "test")
    }


def save_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def evaluate(model: nn.Module, loader: DataLoader, loss_function: nn.Module) -> dict[str, Any]:
    model.eval()
    with nondeterministic_quantized_runtime():
        with torch.no_grad():
            return run_epoch(model, loader, loss_function, None)


@contextmanager
def nondeterministic_quantized_runtime():
    """Allow quantized CPU kernels that lack deterministic implementations."""
    previous = torch.are_deterministic_algorithms_enabled()
    if previous:
        torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        if previous:
            torch.use_deterministic_algorithms(True)


def calibrate(model: nn.Module, loader: DataLoader) -> int:
    model.eval()
    sample_count = 0
    with torch.no_grad():
        for inputs, _ in loader:
            model(inputs)
            sample_count += inputs.shape[0]
    return sample_count


def prepare_float_model(checkpoint: Path) -> QuantizableCNN:
    model = QuantizableCNN()
    weights = torch.load(checkpoint, weights_only=True)
    model.model.load_state_dict(weights)
    return model


def make_static_int8(
    float_model: QuantizableCNN,
    train_loader: DataLoader,
    qat: bool = False,
    qat_epochs: int = 0,
    learning_rate: float = 0.0005,
    loss_function: nn.Module | None = None,
) -> tuple[nn.Module, int]:
    model = copy.deepcopy(float_model)
    model.train(qat)
    model.qconfig = get_default_qat_qconfig("x86") if qat else get_default_qconfig("x86")
    if qat:
        prepare_qat(model, inplace=True)
        assert loss_function is not None
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for _ in range(qat_epochs):
            run_epoch(model, train_loader, loss_function, optimizer)
    else:
        prepare(model, inplace=True)
        calibration_samples = calibrate(model, train_loader)
    model.eval()
    quantized = convert(model, inplace=False)
    return quantized, calibration_samples if not qat else qat_epochs


def main() -> None:
    args = parse_args()
    if args.qat_epochs < 1 or args.batch_size < 1:
        raise ValueError("qat epochs and batch size must be positive")
    seed_everything(args.seed)
    torch.backends.quantized.engine = "x86"
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    loaders = loaders_for(records, args.seed, args.batch_size)
    train_records = [record for record in records if record["split"] == "train"]
    train_counts = Counter(record["class_index"] for record in train_records)
    class_weights = torch.tensor(
        [len(train_records) / (2 * train_counts[index]) for index in (0, 1)],
        dtype=torch.float32,
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    float_model = prepare_float_model(args.checkpoint)
    results: dict[str, dict[str, Any]] = {}

    results["float32"] = {
        "format": "PyTorch float32 state_dict",
        "serialized_state_dict_bytes": save_size_bytes(float_model.model),
        "test": evaluate(float_model.model, loaders["test"], loss_function),
    }

    started = time.perf_counter()
    dynamic_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(float_model.model), {nn.Linear}, dtype=torch.qint8
    )
    results["dynamic_range"] = {
        "format": "dynamic int8 Linear weights; convolution weights remain float32",
        "serialized_state_dict_bytes": save_size_bytes(dynamic_model),
        "test": evaluate(dynamic_model, loaders["test"], loss_function),
        "conversion_seconds": time.perf_counter() - started,
    }

    started = time.perf_counter()
    static_model, calibration_samples = make_static_int8(
        float_model, loaders["train"], loss_function=loss_function
    )
    results["full_int8"] = {
        "format": "post-training static int8",
        "serialized_state_dict_bytes": save_size_bytes(static_model),
        "test": evaluate(static_model, loaders["test"], loss_function),
        "calibration_samples": calibration_samples,
        "conversion_seconds": time.perf_counter() - started,
    }

    started = time.perf_counter()
    qat_model, qat_epochs = make_static_int8(
        float_model,
        loaders["train"],
        qat=True,
        qat_epochs=args.qat_epochs,
        learning_rate=args.learning_rate,
        loss_function=loss_function,
    )
    results["qat"] = {
        "format": "quantization-aware training followed by int8 convert",
        "serialized_state_dict_bytes": save_size_bytes(qat_model),
        "test": evaluate(qat_model, loaders["test"], loss_function),
        "qat_epochs": qat_epochs,
        "conversion_and_training_seconds": time.perf_counter() - started,
    }

    for name, model in (("float32", float_model.model), ("dynamic_range", dynamic_model), ("full_int8", static_model), ("qat", qat_model)):
        torch.save(model.state_dict(), args.output_dir / f"{name}.pt")
    report = {
        "seed": args.seed,
        "candidate": "B depthwise-separable CNN",
        "source_checkpoint": str(args.checkpoint),
        "quantized_engine": torch.backends.quantized.engine,
        "calibration_policy": "train split only",
        "train_sample_count": len(train_records),
        "train_class_counts": dict(sorted(train_counts.items())),
        "class_weights": class_weights.tolist(),
        "results": results,
        "limitations": [
            "The test set contains only two yellow crops; quantized fault recall is not stable evidence.",
            "Candidate B was used as the embedded-size hypothesis, not a selected final health model.",
            "LED color is a proxy for server status, not proof of server health.",
        ],
    }
    output_path = args.output_dir / "quantization_comparison.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"bytes": value["serialized_state_dict_bytes"], "test": value["test"]} for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
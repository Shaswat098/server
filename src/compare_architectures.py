"""Compare the baseline CNN with a depthwise-separable CNN."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from train_baseline import (
    BaselineCNN,
    CropDataset,
    metrics,
    run_epoch,
    seed_everything,
)
from torch.utils.data import DataLoader


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, 1),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class DepthwiseSeparableCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(3, 8),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(8, 16),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(16, 32),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/led_crops/manifest.json"))
    parser.add_argument("--baseline-metrics", type=Path, default=Path("data/interim/baseline/metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/architecture_comparison"))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def train_candidate(
    name: str,
    model: nn.Module,
    split_records: dict[str, list[dict[str, Any]]],
    class_weights: Tensor,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    loaders = {
        split: DataLoader(
            CropDataset(split_records[split]),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed),
        )
        for split in ("train", "valid", "test")
    }
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_path = output_dir / f"{name}.pt"
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], loss_function, optimizer)
        with torch.no_grad():
            validation_metrics = run_epoch(model, loaders["valid"], loss_function, None)
        history.append({"epoch": epoch, "train": train_metrics, "valid": validation_metrics})
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    with torch.no_grad():
        test_metrics = run_epoch(model, loaders["test"], loss_function, None)
    return {
        "name": name,
        "architecture": (
            "depthwise3x3+pointwise1x1 blocks (3->8->16->32), pooling, "
            "global_average_pool, linear(32,2)"
            if name == "candidate_b_depthwise"
            else "see baseline metrics"
        ),
        "parameter_count": parameter_count,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "epochs_completed": len(history),
        "history": history,
        "test": test_metrics,
        "training_time_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "valid", "test")
    }
    train_counts = {index: sum(record["class_index"] == index for record in split_records["train"]) for index in (0, 1)}
    class_weights = torch.tensor(
        [len(split_records["train"]) / (2 * train_counts[index]) for index in (0, 1)],
        dtype=torch.float32,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_b = train_candidate(
        "candidate_b_depthwise",
        DepthwiseSeparableCNN(),
        split_records,
        class_weights,
        args,
        args.output_dir,
    )
    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    comparison = {
        "seed": args.seed,
        "shared_training": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "early_stopping_patience": args.patience,
            "loss": "weighted_cross_entropy",
            "class_weights": class_weights.tolist(),
            "split_manifest": str(args.manifest),
        },
        "candidates": {
            "candidate_a_baseline": {
                "name": baseline["model"]["name"],
                "architecture": baseline["model"]["architecture"],
                "parameter_count": baseline["model"]["parameter_count"],
                "checkpoint_size_bytes": baseline["model"]["checkpoint_size_bytes"],
                "test": baseline["test"],
                "training_time_seconds": baseline["training_time_seconds"],
                "source_metrics": str(args.baseline_metrics),
            },
            "candidate_b_depthwise": candidate_b,
            "candidate_c_transfer_learning": {
                "status": "deferred",
                "reason": "torchvision is not installed and no cached pretrained weights were available; no unverified download was used.",
            },
        },
        "selection": {
            "status": "deferred",
            "reason": "Only two yellow test crops exist, so no candidate can be selected on reliable fault recall.",
        },
    }
    output_path = args.output_dir / "comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "candidate_b": {key: candidate_b[key] for key in ("parameter_count", "checkpoint_size_bytes", "test", "training_time_seconds")}}, indent=2))


if __name__ == "__main__":
    main()
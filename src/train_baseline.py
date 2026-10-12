"""Train and evaluate the small baseline CNN on processed LED crops."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


class CropDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        with Image.open(record["path"]).convert("RGB") as image:
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        return tensor, int(record["class_index"])


class BaselineCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/led_crops/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/baseline"))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def metrics(targets: list[int], predictions: list[int], losses: list[float]) -> dict[str, Any]:
    confusion = [[0, 0], [0, 0]]
    for target, prediction in zip(targets, predictions):
        confusion[target][prediction] += 1
    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    total = len(targets)
    accuracy = (true_negative + true_positive) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    green_recall = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "accuracy": accuracy,
        "precision_yellow": precision,
        "recall_yellow": recall,
        "f1_yellow": f1,
        "recall_green": green_recall,
        "confusion_matrix": confusion,
        "sample_count": total,
        "yellow_count": sum(target == 1 for target in targets),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    targets: list[int] = []
    predictions: list[int] = []
    losses: list[float] = []
    for inputs, labels in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().item()))
        targets.extend(labels.tolist())
        predictions.extend(outputs.argmax(dim=1).tolist())
    return metrics(targets, predictions, losses)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")
    seed_everything(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    split_records = {split: [record for record in records if record["split"] == split] for split in ("train", "valid", "test")}
    train_counts = Counter(record["class_index"] for record in split_records["train"])
    class_weights = torch.tensor(
        [len(split_records["train"]) / (2 * train_counts[index]) for index in (0, 1)],
        dtype=torch.float32,
    )
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
    model = BaselineCNN()
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "baseline_cnn.pt"
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
    elapsed_seconds = time.perf_counter() - started
    checkpoint_bytes = checkpoint_path.stat().st_size
    report = {
        "seed": args.seed,
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "early_stopping_patience": args.patience,
            "loss": "weighted_cross_entropy",
            "class_weights": class_weights.tolist(),
        },
        "model": {
            "name": "Candidate A small custom CNN",
            "architecture": "Conv(3,8)-Conv(8,16)-Conv(16,32)-global_average_pool-linear(32,2)",
            "batch_norm": False,
            "dropout": False,
            "parameter_count": parameter_count,
            "checkpoint_size_bytes": checkpoint_bytes,
        },
        "train_counts": dict(sorted(train_counts.items())),
        "history": history,
        "test": test_metrics,
        "training_time_seconds": elapsed_seconds,
        "limitations": [
            "Validation contains one yellow crop and test contains two yellow crops.",
            "Metrics are not stable estimates of field fault recall.",
            "LED color is a proxy for server status, not proof of server health.",
        ],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(args.output_dir / "metrics.json"), "test": test_metrics, "model": report["model"], "training_time_seconds": elapsed_seconds}, indent=2))


if __name__ == "__main__":
    main()
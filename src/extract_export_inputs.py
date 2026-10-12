"""Extract Candidate B weights and reference logits for TFLite conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from compare_architectures import DepthwiseSeparableCNN
from train_baseline import CropDataset


def main() -> None:
    checkpoint = Path("data/interim/architecture_comparison/candidate_b_depthwise.pt")
    manifest_path = Path("data/processed/led_crops/manifest.json")
    output_dir = Path("data/interim/tflite")
    output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, weights_only=True)
    model = DepthwiseSeparableCNN()
    model.load_state_dict(state)
    model.eval()
    records = [
        record
        for record in json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
        if record["split"] == "test"
    ]
    dataset = CropDataset(records)
    inputs = torch.stack([dataset[index][0] for index in range(len(dataset))])
    with torch.no_grad():
        logits = model(inputs).numpy()
    arrays = {
        key.replace(".", "__"): value.detach().cpu().numpy()
        for key, value in state.items()
    }
    arrays["test_inputs"] = inputs.numpy()
    arrays["reference_logits"] = logits
    np.savez_compressed(output_dir / "candidate_b_export_inputs.npz", **arrays)
    print(f"Wrote {len(records)} test references to {output_dir / 'candidate_b_export_inputs.npz'}")


if __name__ == "__main__":
    main()
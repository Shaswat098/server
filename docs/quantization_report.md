# Phase 5 Quantization Report

## Scope

Quantization was measured on Candidate B, the depthwise-separable CNN selected as the embedded-size hypothesis. All modes used the same processed manifest and test split. Static INT8 calibration used the train split only: 228 crops, including 221 green and 7 yellow. Validation and test images were not used for calibration.

PyTorch 2.5.1 CPU quantization engine: `x86`.

## Results

| Mode | Serialized state dict bytes | Accuracy | Yellow recall | Green recall |
|---|---:|---:|---:|---:|
| Float32 | 9,388 | 0.9672 | 0/2 = 0.0 | 59/59 = 1.0 |
| Dynamic range | 9,958 | 0.9672 | 0/2 = 0.0 | 59/59 = 1.0 |
| Full INT8 | 16,786 | 0.9672 | 0/2 = 0.0 | 59/59 = 1.0 |
| QAT, 20 epochs | 16,786 | 0.9672 | 0/2 = 0.0 | 59/59 = 1.0 |

All modes produced confusion matrix `[[59, 0], [2, 0]]`. Quantization preserved the baseline B predictions but did not improve fault detection.

The full INT8 and QAT serialized state dictionaries are larger than this small float state dictionary because PyTorch serialization includes quantization scales, zero points, packed parameters, and metadata. These are PyTorch artifact sizes, not final TFLite flatbuffer or embedded C-array sizes. Phase 6 must measure the deployment format directly.

## Interpretation

Quantization is technically viable on the available CPU backend and did not change the observed test predictions. It does not repair the central data limitation: only two yellow crops exist in the test set, and both were missed. No quantization mode can be selected as a health-classifier improvement on this evidence.

LED color is a proxy for server status, not proof of actual server health.

## Reproduction

```powershell
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/quantize_model.py
```

Artifacts and machine-readable results are under `data/interim/quantization/`.

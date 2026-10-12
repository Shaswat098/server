# Phase 6 TFLite Export and Desktop Verification

## Export

The workspace `.venv` now provides TensorFlow CPU 2.21.0. Candidate B was mapped from its PyTorch checkpoint into an equivalent Keras graph:

- Input: float32 NHWC `[None, 64, 64, 3]`
- Pixel range: `[0.0, 1.0]`
- Three depthwise 3x3 plus pointwise 1x1 blocks
- Max pooling after the first two blocks
- Global average pooling and two-logit dense head

The exported float32 TFLite flatbuffer is 9,848 bytes.

## Desktop parity

The export workflow generated PyTorch reference logits for all 61 test crops, then compared them with both Keras and the TFLite interpreter:

- Keras vs. PyTorch maximum absolute logit error: `5.96e-08`
- TFLite vs. PyTorch maximum absolute logit error: `1.79e-07`
- Test samples checked: 61

This verifies the tensor-axis mapping, RGB-to-NHWC conversion, resize contract, model weights, and TFLite interpreter output on desktop.

## Reproduction

The system PyTorch interpreter creates the neutral export bundle:

```powershell
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/extract_export_inputs.py
```

The TensorFlow workspace environment converts and verifies it:

```powershell
& '.venv/Scripts/python.exe' src/export_tflite.py
```

Artifacts are under `data/interim/tflite/` and the machine-readable parity result is `desktop_verification.json`.

## Boundary

This phase produced and verified a float32 TFLite model. The PyTorch INT8 artifacts from Phase 5 are not automatically equivalent to a TFLite INT8 flatbuffer. Representative-dataset TFLite INT8 conversion and embedded parity remain deployment work for later phases.

The test set has only two yellow crops, so parity confirms implementation equivalence, not health-classifier quality. LED color is a proxy for server status, not proof of actual server health.

# Final Evaluation

## Executive result

The software pipeline is reproducible from raw COCO data through a desktop-verified float32 TFLite artifact. It is not ready to be called a reliable server-health classifier or ESP32 deployment.

## Measured results

- Raw images: 104
- Derived binary crops: 339 (329 green, 10 yellow)
- Candidate A: 6,098 parameters, test accuracy 0.8852, yellow recall 0/2
- Candidate B: 1,056 parameters, test accuracy 0.9672, yellow recall 0/2
- All four quantization modes: identical test confusion matrix `[[59, 0], [2, 0]]`
- Float32 TFLite size: 9,848 bytes
- Full-integer INT8 TFLite size: 9,720 bytes; ESP32 firmware image: 1,105,744
  bytes
- TFLite/PyTorch max absolute logit error: `1.79e-07` over 61 test crops

## Decisions

Candidate B remains a deployment-size hypothesis, not a selected health model. Accuracy is not an acceptable selection metric here because it masks missed yellow faults. No final model selection is made until more yellow examples are collected and a leakage-aware device/session evaluation is possible.

## Open work

Phase 7 float32 and INT8 hardware integration are complete on the connected
Robodo ESP32-CAM. Float32 parity passed at 586,318 microseconds mean invoke;
INT8 completed all 61 parity samples at a one-count raw-output tolerance and
measured 592,041 microseconds mean invoke. Both use a 256 KiB PSRAM arena.
OV2640 capture and camera-derived float32 inference passed. Camera preprocessing
equivalence is still unvalidated, although the INT8 camera path now also
captures and infers successfully at 592,946 microseconds in the measured run.
Blue handling and abstention behavior remain open.

## Reproduction commands

```powershell
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/run_eda.py
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/prepare_dataset.py
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/train_baseline.py
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/compare_architectures.py
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/quantize_model.py
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/extract_export_inputs.py
& '.venv/Scripts/python.exe' src/export_tflite.py
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/create_embedded_bundle.py
```

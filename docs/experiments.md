# Experiment Index

| Phase | Experiment | Evidence |
|---|---|---|
| 0 | Environment and raw COCO audit | `docs/dataset_card.md` |
| 1 | Dimensions, balance, duplicates, bbox geometry, RGB diagnostic | `data/interim/eda_report.json`, `docs/eda_report.md` |
| 2 | Padded crop creation and source-image grouped split | `data/processed/led_crops/manifest.json`, `docs/data_pipeline.md` |
| 3 | Candidate A baseline CNN | `data/interim/baseline/metrics.json`, `docs/baseline_report.md` |
| 4 | Candidate A vs. depthwise Candidate B | `data/interim/architecture_comparison/comparison.json`, `docs/architecture_comparison.md` |
| 5 | Float32, dynamic, INT8, and QAT comparison | `data/interim/quantization/quantization_comparison.json`, `docs/quantization_report.md` |
| 6 | PyTorch-to-Keras-to-TFLite mapping and desktop parity | `data/interim/tflite/desktop_verification.json`, `docs/tflite_report.md` |
| 7 | Embedded bundle and TFLite Micro skeleton | `embedded/`, `docs/phase7_blocker.md` |
| 8 | Final review and software handoff | `docs/final_evaluation.md`, `docs/model_card.md`, `docs/deployment.md` |

All reported metrics are measured artifacts. The two-yellow test count prevents reliable model selection for fault recall.

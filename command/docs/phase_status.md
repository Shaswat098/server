# Phase Status

Do not start a phase until the previous one is `Done`, unless explicitly told to skip ahead.

| Phase | Name | Status | Notes |
|---|---|---|---|
| 0 | Environment check + dataset audit + project skeleton | Done | Verified Python 3.11, CPU PyTorch 2.5.1, 104 JPEGs, COCO categories, and annotation counts. Raw export preserved. |
| 1 | EDA (class balance, dims, duplicates, leakage risk, separability) | Done | Measured dimensions, class balance, bbox geometry, exact duplicates, filename overlap, and coarse bbox RGB diagnostics. Yellow is absent from valid; near-duplicate/device leakage remains unresolved. |
| 2 | Data pipeline: cleaning, blue exclusion, leakage-safe split | Done | Derived 339 RGB 64x64 crops; blue excluded; source-image grouped train/valid/test split with 7/1/2 yellow crops. Device/session grouping remains unavailable. |
| 3 | Baseline CNN training (Candidate A) | Done | Trained deterministic 6,098-parameter CNN. Test yellow recall was 0/2; results are feasibility evidence only because fault counts are tiny. |
| 4 | Architecture comparison (A vs. depthwise-separable B vs. transfer learning C) + selection | Done | Compared A and B under identical settings. B is 1,056 parameters but both models had 0/2 yellow test recall. C deferred: torchvision and cached weights unavailable; no final selection made. |
| 5 | Quantization (fp32 / dynamic / INT8 / QAT comparison) | Done | Compared all four modes on Candidate B with train-only calibration (228 samples). All had 0/2 yellow test recall; deployment-format size remains for Phase 6. |
| 6 | TFLite conversion + desktop verification | Done | Exported and verified float32 TFLite Candidate B. Max TFLite/PyTorch logit error 1.79e-07; flatbuffer 9,848 bytes. INT8 TFLite remains deployment work. |
| 7 | Embedded integration (TFLite Micro / ESP) + desktop-vs-embedded parity test | Blocked | Generated model_data.h and 61-sample parity fixture; no ESP toolchain or serial Robodo ESP32-CAM available for compilation/on-device measurements. |
| 8 | Final evaluation report + docs polish + decision log review | Done | Added final evaluation, model card, deployment handoff, experiment index, and corrected stale blocker/readme text. Phase 7 remains hardware-blocked by explicit user approval. |

Update the `Status` (`Not started` / `In progress` / `Blocked` / `Done`) and `Notes` columns whenever a phase starts or finishes.

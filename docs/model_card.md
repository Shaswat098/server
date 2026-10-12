# Model Card

## Intended use

This project investigates a small image model for a server LED color proxy on a Robodo ESP32-CAM. It is not evidence of actual server health.

## Outputs

- `0`: green/healthy proxy
- `1`: yellow/fault proxy
- Blue is excluded from binary training, but the desktop policy detects visible
	blue and maps it to output `1` (`yellow/fault`).

## Candidate status

No final health classifier is selected. Candidate B is the current deployment-size hypothesis because it has 1,056 parameters, but it missed both yellow test crops. Candidate C transfer learning was not run because verified pretrained tooling and weights were unavailable.

## Data

The source export contains 104 images and 390 annotations. The derived data contains 339 green/yellow crops: 329 green and 10 yellow. The source-image grouped test split contains 59 green and 2 yellow crops. Device/session grouping is unavailable, so near-duplicate scene leakage cannot be ruled out.

## Results

Candidate B test accuracy is 0.9672, green recall is 1.0, and yellow recall is 0/2 = 0.0. This accuracy is misleading because the test set is overwhelmingly green. Float32, dynamic-range, full INT8, and QAT variants all had the same `[[59, 0], [2, 0]]` test confusion matrix.

## Limitations and risks

- Yellow data is too scarce for stable fault-recall estimates.
- The provided source validation set had no yellow annotations.
- The blue override is a heuristic, not a trained detector. Blue non-LED
	objects or full scenes can cause false unhealthy results.
- Hardware latency, memory, camera preprocessing, and TFLite Micro compatibility are unmeasured.
- LED color is hardware/configuration dependent and only a proxy for server status.

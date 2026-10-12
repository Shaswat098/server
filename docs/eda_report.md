# Phase 1 EDA Report

## Dataset shape

The audit was run from the raw COCO export with `src/run_eda.py`.

| Split | Images | Annotations | Images without annotations |
|---|---:|---:|---:|
| train | 94 | 354 | 0 |
| valid | 10 | 36 | 0 |
| Total | 104 | 390 | 0 |

All 104 images are RGB JPEGs at 640x480.

## Annotation balance

| Split | Blue | Green | Yellow |
|---|---:|---:|---:|
| train | 41 | 303 | 10 |
| valid | 10 | 26 | 0 |
| Total | 51 | 329 | 10 |

The binary task maps green to 0 and yellow to 1. Blue remains excluded from binary loss. Yellow is severely underrepresented and absent from the provided validation annotations, so the source validation split cannot measure fault recall.

## Bounding boxes

Train bounding boxes range from 5.53 to 37.48 pixels wide and 6.12 to 37.61 pixels high. Validation boxes range from 8.45 to 28.78 pixels wide and 7.57 to 30.23 pixels high. This confirms that the LED regions are small relative to the 640x480 frames and that crop/resizing policy will materially affect the task.

## Duplicate and leakage checks

- Exact SHA-256 duplicate groups: 0
- Repeated source filename stems across images/splits: 0
- Cross-split filename overlap: 0

These checks reduce evidence for exact duplication but do not prove that near-duplicate views, repeated devices, or recording sessions are independent. The filenames contain no reliable device/session metadata. Phase 2 must document its split strategy and, if no grouping metadata can be recovered, state that limitation explicitly.

## Coarse color separability diagnostic

Mean RGB values inside annotation boxes were computed as a descriptive diagnostic:

| Split | Label | Count | Mean R | Mean G | Mean B |
|---|---|---:|---:|---:|---:|
| train | blue | 41 | 80.13 | 121.18 | 182.08 |
| train | green | 303 | 104.72 | 133.19 | 87.09 |
| train | yellow | 10 | 155.91 | 150.32 | 107.82 |
| valid | blue | 10 | 88.02 | 131.11 | 195.02 |
| valid | green | 26 | 110.20 | 131.28 | 89.89 |

The means are directionally consistent with the label names, but this is not evidence of robust separability: it ignores background, illumination, saturation, crop alignment, device variation, and the absent validation-yellow class. A model-independent separability result requires more yellow examples and a leakage-aware evaluation design.

## Phase 1 conclusion

The dataset is usable for continued investigation, but not yet for trustworthy binary model selection. The controlling risks are yellow scarcity, absent validation faults, and unknown device/session grouping. Phase 2 should derive LED crops, exclude blue explicitly, and define a leakage-aware split before training.

LED color remains a proxy for server status, not proof of actual server health.

# Phase 2 Data Pipeline

## Derived dataset

`src/prepare_dataset.py` reads the immutable COCO export and writes `data/processed/led_crops/`.

- One derived crop per green or yellow COCO annotation.
- Blue annotations are excluded from the binary dataset; they are never relabeled.
- Images containing both blue and a binary label retain only the green/yellow crops.
- Blue-only source images are excluded entirely: 13 images.
- Bounding boxes receive 25% padding based on the larger bbox dimension.
- Crops are converted to RGB and resized to 64x64 with LANCZOS.
- No augmentation is applied.
- `manifest.json` records source image, annotation id, original bbox, split, label, and preprocessing settings.

## Split

The source train/valid directories are not reused as model splits because source valid contains no yellow annotations. All eligible source images are regrouped deterministically with seed `20260914`:

| Split | Source images | Crops | Green | Yellow |
|---|---:|---:|---:|---:|
| train | 63 | 228 | 221 | 7 |
| valid | 14 | 50 | 49 | 1 |
| test | 14 | 61 | 59 | 2 |

All crops from one source image remain in one split. This prevents frame-level crop leakage. The dataset has no device/session metadata, so repeated physical devices or recording sessions cannot be ruled out.

## Verification

From the workspace root:

```powershell
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/prepare_dataset.py
```

Then inspect `data/processed/led_crops/manifest.json`. The raw export under `data/raw/` must remain unchanged.

## Limitations

Ten yellow annotations are insufficient for a stable estimate of fault recall. The test set contains only two yellow crops, so later model metrics must show counts and uncertainty rather than being treated as definitive. LED color remains a proxy for server status, not proof of actual server health.

# Dataset Card

## Source

- Dataset: `led_lights-6djfb`, Roboflow Universe, v2
- Local path: `data/raw/led_lights_coco/`
- License reported by source README: CC BY 4.0
- Export format: COCO detection
- Source README reports 104 images

## Verified local audit

- Train images: 94
- Validation images: 10
- Total JPEG images: 104
- COCO categories in both files: `leds` (id 0), `blue` (id 1), `green` (id 2), `yellow` (id 3)
- Train annotations: blue 41, green 303, yellow 10
- Validation annotations: blue 10, green 26, yellow 0
- Category id 0 is present in metadata but has no annotations in the checked splits

## Interpretation

The validation split has no yellow annotations, so it is not suitable for measuring fault recall as-is. Phase 1 must inspect dimensions, bounding boxes, duplicates, and possible device/session grouping before any split is changed. Raw files must remain untouched.

Blue is a locate/identify indicator and is excluded from binary health training. This creates an explicit open deployment issue: the binary model has no blue output or abstention behavior yet.

LED color is a proxy for server status, not proof of real server health; hardware/configuration semantics are not established by this dataset.

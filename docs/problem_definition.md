# Problem Definition

## Objective

Investigate a binary image classifier for a server LED health proxy:

- `0`: green, healthy
- `1`: yellow, fault

Blue indicates server locate/identify mode, not health. Blue images or regions are excluded from the binary training loss and must not be silently relabeled.

## Scope and limitation

The model predicts an LED-color proxy, not actual server health. LED semantics depend on hardware and configuration, so deployment decisions require field validation.

The source export is COCO object detection data. A later phase must determine whether object crops provide a valid classifier input and whether multiple views of one device or session require grouped splitting.

## Target

Confirmed hardware: Robodo ESP32-CAM, AI-Thinker-style ESP32 module with OV2640 camera. Exact usable SRAM, latency, and flash/RAM budgets remain unverified on-device.

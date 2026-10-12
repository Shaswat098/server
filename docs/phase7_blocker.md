# Phase 7 Hardware Record

Phase 7 float32 and INT8 hardware integration is complete on the connected
Robodo/AI-Thinker-style ESP32-CAM. Windows enumerated the CH340 adapter as
`COM8`; `COM7` was not present in the device list.

Completed:

- Generated the verified 9,848-byte float32 TFLite model array.
- Built and flashed PlatformIO firmware with
  `sebastianregelmann/tflite-micro@1.26.41`.
- Registered the six operators used by Candidate B.
- Allocated and measured a 256 KiB tensor arena in PSRAM.
- Passed all 61 desktop parity samples on-device at `1e-4` tolerance.
- Captured a 160x120 RGB565 OV2640 frame and ran camera-derived inference.

Measured:

- Chip: `ESP32-D0WD-V3`, 240 MHz
- Flash: 4,194,304 bytes
- Firmware image: 3,424,640 bytes
- Application partition: 4,063,232 bytes
- Free heap after tensor allocation: 336,916 bytes
- Free PSRAM after tensor allocation: 3,929,963 bytes
- Mean model invoke latency: 586,318 microseconds across 61 samples
- Camera-derived invoke latency: 585,931 microseconds in the captured run

INT8 measurement:

- Input/output tensor types: int8 (`9`)
- Firmware image: 1,105,744 bytes
- Free heap after tensor allocation: 346,264 bytes
- Free PSRAM after tensor allocation: 3,929,963 bytes
- All 61 parity samples completed; maximum raw output difference was one
  count, accepted at the explicit tolerance of one int8 count
- Mean model invoke latency: 592,041 microseconds across 61 samples

Remaining deployment work is camera-specific preprocessing validation and an
INT8 camera path. The
camera path currently uses a centered crop and nearest-neighbor resize; it
must not be described as equivalent to the training-time annotated crop and
Lanczos preprocessing.

LED color remains a proxy for server status, not proof of actual server health.

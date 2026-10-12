# TFLite Micro Bundle

`model_data.h` contains the verified 9,848-byte Candidate B float32 TFLite flatbuffer. `parity_fixture.h` contains 61 desktop test inputs in NHWC float32 order and their PyTorch reference logits.

`tflite_micro_inference.cpp` is the reusable integration boundary. The
PlatformIO firmware in `src/main.cpp` runs the measured hardware parity and
camera checks. Build and flash from the repository root:

```powershell
.\.venv\Scripts\pio.exe run -d embedded -t upload --upload-port COM8
```

The measured float32 firmware uses:

1. TFLite Micro package `sebastianregelmann/tflite-micro@1.26.41`.
2. A 256 KiB tensor arena allocated in PSRAM.
3. Depthwise convolution, convolution, ReLU, max pool, mean/global average
   pool, and fully connected operators.
4. The 61-sample desktop parity fixture.

Measured results are recorded in `docs/deployment.md`. The separate
`esp32cam_int8` environment uses the full-integer model in `int8_bundle/`,
int8 input/output tensors, and the same 256 KiB PSRAM arena. Its camera path
also uses a centered 120x120 crop and nearest-neighbor resize. Both camera
paths are integration demonstrations, not validated equivalents of the
annotated training crop pipeline.

The INT8 firmware emits one length-framed software-JPEG image during startup
between `camera_jpeg_begin:<length>` and `camera_jpeg_end`. Capture it from the
project root
with:

```powershell
& '.\.venv\Scripts\python.exe' src\capture_camera_frame.py `
  --output data\interim\camera_captures\green_candidate.jpg
```

Place one known LED color in the camera view before running the command.
Repeat separately for green, yellow, and blue, using descriptive output names.
The capture uses 921600 baud and is evidence for crop and channel-policy
review; it must not be added to the training set without a documented split
decision.

## Standalone livestream color-policy classifier

The `esp32cam_stream` environment is a separate camera diagnostic; it does
not run the TFLite CNN. It continuously analyzes the currently displayed
livestream in the browser using the documented color policy. Build and upload
it with:

```powershell
.\.venv\Scripts\pio.exe run -d embedded -e esp32cam_stream -t upload --upload-port COM8
```

After the board joins the configured 2.4 GHz network, open the IP address
printed in the serial monitor. The page displays the MJPEG livestream and
re-evaluates it every 500 ms:

- `SERVER STATUS: WORKING / GREEN` means the green policy matched at least 1%
  of visible pixels.
- `SERVER STATUS: FAULT / YELLOW` means the yellow policy matched at least 1%
  of visible pixels.
- `BLUE LOCATE INDICATOR - HEALTH NOT CLASSIFIED` keeps blue separate from
  health labels.
- `UNKNOWN / INSUFFICIENT COLOR SIGNAL` avoids guessing when no color is
  sufficiently visible.

The blue rule is the same policy used by `src/classify_image.py`:
`blue > 80`, blue dominant over red/green, and `blue - red > 20`.
The green/yellow rules are simple diagnostic pixel policies, not the trained
TFLite CNN. They must be calibrated against known LED captures before field
use. LED color is a proxy for server status, not proof of real server health.

# Deployment Handoff

## Verified software artifact

- Model: Candidate B depthwise-separable CNN
- Format: float32 TFLite
- Flatbuffer size: 9,848 bytes
- Input: RGB/NHWC float32, shape `64x64x3`, values normalized to `[0, 1]`
- Output: two float32 logits, index 0 green and index 1 yellow
- Desktop TFLite/PyTorch maximum absolute logit error: `1.79e-07`

The flatbuffer is at `data/interim/tflite/candidate_b_float32.tflite`. The TFLite Micro C array is at `embedded/model_data.h`.

## Embedded handoff

`embedded/tflite_micro_inference.cpp` provides the integration boundary and `embedded/parity_fixture.h` provides 61 desktop reference cases. The live PlatformIO firmware is in `embedded/src/main.cpp`.

## Measured ESP32-CAM result

Measured on the connected Robodo/AI-Thinker-style ESP32-CAM through CH340
`COM8`, chip `ESP32-D0WD-V3`, 240 MHz:

- TFLite Micro package: `sebastianregelmann/tflite-micro@1.26.41`
- Model: verified Candidate B float32 TFLite
- Tensor arena: 262,144 bytes allocated in PSRAM
- Free PSRAM before allocation: 4,192,123 bytes
- Free PSRAM after allocation: 3,929,963 bytes
- Free heap after allocation: 336,916 bytes
- Flash chip: 4,194,304 bytes
- Firmware image: 3,424,640 bytes; custom single-app partition uses 4,063,232 bytes
- Desktop parity: 61/61 samples passed at tolerance `1e-4`
- Mean TFLite Micro invoke latency over 61 samples: 586,318 microseconds
- Camera: `ESP_OK`, 160x120 RGB565 frame, 38,400 bytes
- Camera-derived inference: `ESP_OK`, 585,931 microseconds in the captured run

## INT8 deployment candidate

A separate full-integer TFLite artifact was generated with calibration from
the 228 training crops only:

- File: `data/interim/tflite/candidate_b_int8.tflite`
- Size: 9,720 bytes
- Input: int8, scale `0.0039215689`, zero point `-128`
- Output: int8, scale `0.0025411167`, zero point `-39`
- Test accuracy: 96.72%
- Yellow precision: 0.00%
- Yellow recall: 0/2
- Confusion matrix: `[[59, 0], [2, 0]]`
- Maximum dequantized logit delta versus float32 TFLite: `0.00188455`

The embedded C array is in `embedded/int8_bundle/model_data.h`. This artifact
- was flashed and measured on the same board. Tensor input/output types were
  both int8 (`9` in the TFLite Micro enum).
- Tensor allocation succeeded with the same 262,144-byte PSRAM arena.
- Free PSRAM after allocation: `3,929,963` bytes; free heap after allocation:
  `346,264` bytes.
- Embedded parity completed all 61 samples. Seven raw output values differed
  from the desktop fixture by one quantization count; the maximum difference
  was `1`, so parity passed at an explicit tolerance of one int8 count.
- Mean TFLite Micro invoke latency over 61 samples: `592,041` microseconds.
- INT8 firmware image: `1,105,744` bytes, versus `3,424,640` bytes for the
  float32 firmware image.
- INT8 camera capture: `ESP_OK`, 160x120 RGB565 frame, 38,400 bytes.
- INT8 camera-derived inference: `ESP_OK`, `592,946` microseconds in the
  captured run; the sample was predicted as class 0 with dequantized logits
  green `0.419284` and yellow `-0.226159`.

The INT8 model is therefore much smaller in flash but was not faster than the
float32 model on this ESP32 run. The measured result does not yet establish
lower arena use or equivalent camera behavior. The camera run used the same
centered 120x120 crop and nearest-neighbor resize as the float32 integration
demonstration, with input quantization scale `0.0039215689` and zero point
`-128`.

The camera path currently takes a centered 120x120 crop from the 160x120 frame,
converts RGB565 to RGB float32, and uses nearest-neighbor resize to 64x64. This
is a working hardware path, but it is not yet proven equivalent to the
training-time annotated crop and Lanczos preprocessing. The camera prediction
must therefore be treated as an integration demonstration, not a validated
field-health result.

Remaining:

- Validate camera channel order, centered crop, resize, and normalization
  against representative LED captures.
- The desktop policy maps sufficiently visible blue pixels to class 1
  (`yellow/fault`); this is a heuristic and must be calibrated against camera
  images before field deployment.
- Validate the shared camera crop, channel order, resize, and normalization
  against representative LED captures before treating either camera result as
  a field-health measurement.

  ## Standalone livestream color-policy classifier

  The separate `esp32cam_stream` firmware keeps the camera livestream isolated
  from the classifier firmware. Because the OV2640 configuration uses RGB565
  capture and software JPEG conversion, the browser can inspect the same
  displayed livestream frame without relying on the unreliable raw serial-frame
  path or opening a second camera capture. The page draws the current MJPEG
  image to a hidden canvas every 500 ms and applies the color-policy classifier.

  The blue policy exactly matches `src/classify_image.py`: `blue > 80`,
  `blue > red * 1.2`, `blue > green * 1.1`, and `blue - red > 20`, with at
  least 1% of visible pixels satisfying it. The diagnostic green policy is
  `green > 80`, `green > red * 1.15`, `green > blue * 1.15`, and
  `green - red > 20`. The yellow policy requires `red > 80`, `green > 80`,
  both red and green to exceed blue by 20%, and `abs(red - green) < 90`.

  The page reports working/green, fault/yellow, blue locate indicator, or
  unknown/insufficient signal and shows the measured fractions. This is a
  continuous browser-side color-policy diagnostic, not TFLite CNN inference.
  It does not add blue to the binary health model, and LED color remains a
  hardware/configuration-dependent proxy rather than proof of real server
  health. These thresholds are not yet validated across devices or lighting
  conditions.

  The current stream handler is a single blocking MJPEG connection. For rover
  operation, use the Streamlit dashboard as the only client of
  `http://<ESP32-IP>/stream`; close the direct ESP32 browser stream first.
  Otherwise a second client may time out while the first client owns the camera
  stream. Supporting multiple simultaneous viewers is a separate server
  architecture change and is not required for the rover-to-laptop path.

  ## Laptop rover dashboard

  For rover operation, the ESP32-CAM only needs power after the stream firmware
  has been flashed. It joins the configured 2.4 GHz Wi-Fi network and exposes
  `/stream`; the laptop connects to that URL over the network. The laptop
  dashboard is `src/streamlit_dashboard.py`. It parses the MJPEG stream,
  classifies each received JPEG with the same green/yellow/blue policy as the
  ESP32 livestream page, and displays the frame and status.

  Install and run:

  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
  .\.venv\Scripts\python.exe -m streamlit run src\streamlit_dashboard.py -- `
    --stream-url http://<ESP32-IP>/stream
  ```

  The dashboard is intentionally color-policy based. It does not claim that the
  small CNN has been validated for live health detection: the recorded test set
  had only two yellow samples and yellow recall was `0/2`. Known-color camera
  captures, camera ROI equivalence, and lighting calibration remain the
  incomplete hardware-validation work.

## Camera preprocessing contract check

The deterministic desktop contract check is implemented in
`src/validate_camera_preprocessing.py` and writes
`data/interim/camera_preprocessing_validation.json`. It verifies that the
documented 160x120 frame bounds, centered 120x120 crop, floor-index nearest
resize, RGB565 channel extraction, and int8 input quantization produce a
64x64x3 int8 tensor without out-of-range values.

This check passed, including crop bounds and tensor shape. It does not validate
that the LED is inside the crop, nor does it establish equivalence with the
training-time padded crop and Lanczos resize. A representative captured frame
fixture is still required for that decision.

The INT8 firmware emits one length-framed software-JPEG image at startup
between `camera_jpeg_begin:<length>` and `camera_jpeg_end`. The host decoder
`src/capture_camera_frame.py` receives it at 921600 baud and saves it as a
JPEG. This avoids raw RGB565 serial synchronization and decoding ambiguity:

```powershell
& '.\.venv\Scripts\python.exe' src\capture_camera_frame.py `
  --output data\interim\camera_captures\green_candidate.jpg
```

Repeat for known green, yellow, and blue LED scenes with descriptive output
names. These captures are for crop and preprocessing evidence, not additional
training data.

The repaired transport was verified with
`data/interim/camera_captures/fixed_capture_20261012_155533.jpg`. It decoded
to 160x120, 1,019 JPEG bytes, and a mean brightness of `247.14/255`. The
brightest region was detected at x=`48..159`, y=`47..78`; 82.1% of the
brightest pixels were inside the fixed crop. This proves the earlier black
PNG was a raw RGB565 serial transport problem, not a dead camera.

Use `src/analyze_camera_capture.py` to quantify brightness, the brightest
region, color dominance, and how much of the bright region falls inside the
fixed crop:

```powershell
& '.\.venv\Scripts\python.exe' src\analyze_camera_capture.py `
  data\interim\camera_captures\green_candidate.png `
  --output data\interim\camera_captures\green_candidate_analysis.json
```

## Desktop upload demo

The model can also be exercised locally with the upload demo:

```powershell
.\.venv\Scripts\Activate.ps1
python src\classify_image.py
```

This opens a file picker. Select a close-up crop containing one LED. The script
uses RGB conversion, Lanczos resize to 64x64, and float32 division by 255,
which matches the verified model contract. It reports class 0 as
`green/healthy` and class 1 as `yellow/fault`. A full server scene is not
supported because the current model is a classifier rather than a detector.
Blue remains outside the binary training output space, but the desktop field
policy maps detected blue to `yellow/fault`. LED color is a proxy for server
status, not proof of real server health.

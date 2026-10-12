# Lightweight Server LED Status Classifier

This project investigates a small image classifier for server LED status on a Robodo ESP32-CAM target.

## Current phase

Software phases 0 through 6 and the Phase 8 documentation review are complete.
Phase 7 float32 and INT8 hardware integration/parity are complete on the
connected ESP32-CAM; camera preprocessing validation remains open. See
[docs/deployment.md](docs/deployment.md).

## Dataset

The raw Roboflow Universe export is kept unchanged at `data/raw/led_lights_coco/`. It contains 104 JPEG images and COCO annotations in `train/` and `valid/`.

The verified COCO categories are `blue`, `green`, and `yellow`. The project task is binary health classification: green/healthy = 0 and yellow/fault = 1. Blue is a locate/identify indicator and must be excluded from the binary training loss, never silently relabeled.

The raw export is detection-oriented. The completed EDA found that the provided validation annotations contain no yellow objects, so a deterministic source-image grouped split was derived for the binary experiments. The resulting test set still contains only two yellow crops.

LED color is only a proxy for server status; hardware and configuration semantics must be verified separately.

## Measured model validation

The current deployment-size candidate is **Candidate B**, a 1,056-parameter
depthwise-separable CNN. The split is source-image grouped and contains 228
training crops, 50 validation crops, and 61 test crops. The validation set has
49 green and only 1 yellow crop; the test set has 59 green and only 2 yellow
crops.

### Candidate B metrics

| Split | Loss | Accuracy | Yellow precision | Yellow recall | Yellow F1 | Green recall | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| Validation (epoch 60) | 0.6266 | 98.00% | 0.00% | 0.00% | 0.00% | 100.00% | `[[49, 0], [1, 0]]` |
| Test | 0.6370 | 96.72% | 0.00% | 0.00% | 0.00% | 100.00% | `[[59, 0], [2, 0]]` |

The confusion matrix uses rows as actual classes and columns as predicted
classes, in the order `[green, yellow]`. Therefore, on the test set the model
correctly classified 59 green crops, missed both yellow crops, and produced no
yellow predictions. This means the apparently high accuracy does **not**
demonstrate reliable fault detection. Overall/macro precision and F1 are not
reported because the stored evaluation tracks the safety-critical yellow
metrics directly.

### Architecture comparison

| Candidate | Parameters | Checkpoint size | Test accuracy | Yellow precision | Yellow recall | Yellow F1 | Green recall | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A custom CNN | 6,098 | 27,772 bytes | 88.52% | 0.00% | 0.00% | 0.00% | 91.53% | 15.26 s |
| B depthwise-separable CNN | 1,056 | 9,832 bytes | 96.72% | 0.00% | 0.00% | 0.00% | 100.00% | 18.76 s |

Candidate B is carried forward as a deployment-size hypothesis, not as a
validated final health model. Model selection is deferred until more yellow
examples and device/session-diverse data are collected.

### Quantization comparison

All four measured Candidate B PyTorch modes produced the same test metrics:
96.72% accuracy, 0.00% yellow precision, 0.00% yellow recall, 0.00% yellow
F1, and 100.00% green recall, with confusion matrix `[[59, 0], [2, 0]]`.
These are serialized PyTorch artifact sizes, not final TFLite sizes:

| Mode | Serialized size |
|---|---:|
| Float32 | 9,388 bytes |
| Dynamic range | 9,958 bytes |
| Full INT8 | 16,786 bytes |
| QAT | 16,786 bytes |

The verified float32 TFLite flatbuffer is 9,848 bytes. Its maximum absolute
logit difference from the PyTorch reference was `1.79e-07` over 61 test crops.
These results verify conversion parity, not real-world health-classifier
quality.

An additional full-integer TFLite candidate is available at
`data/interim/tflite/candidate_b_int8.tflite`. It is 9,720 bytes, uses int8
input/output, was calibrated from 228 training samples only, and has a maximum
dequantized logit delta of `0.00188455` versus float32 TFLite. Its test metrics
remain 96.72% accuracy with 0/2 yellow recall. On the ESP32 it completed all
61 parity samples with a maximum one-count raw-output difference at the
documented tolerance, but averaged 592,041 microseconds per invoke, compared
with 586,318 microseconds for float32. The INT8 firmware image is 1,105,744
bytes versus 3,424,640 bytes for float32.

**Evaluation limitation:** the yellow class has only 1 validation crop and 2
test crops. Precision, recall, and F1 for fault detection are therefore
unstable but important: missing a yellow fault is more serious than raising a
false alarm. Blue is excluded from the binary task and is not an output class.
LED color remains a proxy for server status, not proof of actual server health.

## Layout

- `data/raw/`: immutable source export
- `data/interim/`: derived audit and intermediate artifacts
- `data/processed/`: leakage-safe model inputs created in a later phase
- `configs/`: reproducible experiment settings
- `src/`: project code
- `docs/`: dataset, model, deployment, and decision records

## Try a local image

The verified desktop model can classify a close-up LED crop before hardware
deployment. From the project root, use the workspace virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install Pillow
python src\classify_image.py
```

Select **Upload image**, choose a JPG or PNG showing one LED close up, and read
the prediction. The model resizes the image to 64x64 RGB and scales pixels to
`[0, 1]`, matching the TFLite contract. A full server photograph is not
supported because this model does not include an object detector; crop the LED
first. CLI use is also available:

```powershell
python src\classify_image.py --image path\to\led_crop.jpg
```

The outputs are `green/healthy` (class 0) and `yellow/fault` (class 1). Blue is
not a trained output class and may be forced into one of those two results.
LED color is a proxy for server status, not proof of actual server health.

## Rover livestream dashboard

The ESP32-CAM stream firmware is separate from the TFLite firmware. It sends
RGB565 camera frames over Wi-Fi and converts them to an MJPEG stream. Flash it
from the repository root:

```powershell
.\.venv\Scripts\pio.exe run -d embedded -e esp32cam_stream -t upload --upload-port COM8
```

After the ESP32 joins the same 2.4 GHz Wi-Fi network as the laptop, read its IP
from the serial monitor:

```powershell
.\.venv\Scripts\pio.exe device monitor -d embedded -e esp32cam_stream -p COM8 -b 115200
```

The ESP32 stream itself starts automatically after flashing. Its URLs are
`http://<ESP32-IP>/` and `http://<ESP32-IP>/stream`; it does not require a
USB connection after flashing. The rover can carry the ESP32-CAM while the
laptop receives the stream over Wi-Fi.

Install the dashboard dependency once:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
```

Start the laptop dashboard, replacing the IP with the address printed by the
ESP32:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\streamlit_dashboard.py -- `
  --stream-url http://10.20.134.230/stream
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`. The dashboard continuously displays the stream and
classifies visible green, yellow, and blue pixels. It reports green as
working, yellow as fault, blue as locate/identify with health unclassified,
and ambiguous frames as unknown. This is the same explicit color policy as
the ESP32 livestream page; it is not TFLite CNN inference.

Use the laptop dashboard as the only stream client during operation. The
current ESP32 Arduino `WebServer` stream handler is blocking and the board has
one camera framebuffer, so opening the direct ESP32 stream page and the
dashboard at the same time can make the second client time out. Close the
direct ESP32 stream page before starting the dashboard.

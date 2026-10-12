# Architecture Decision Records

## ADR-0001: Preserve the Roboflow COCO export as raw input

- Date: 2026-09-14
- Status: Accepted
- Context: The downloaded dataset contains 104 images and COCO annotations in train/valid directories.
- Decision: Treat `data/raw/led_lights_coco/` as immutable. Derive all audit, crop, split, and preprocessing artifacts under `data/interim/` or `data/processed/`.
- Why: Preserves provenance and makes preprocessing reproducible.
- Alternatives considered: Editing annotations in place; rejected because it destroys the source record.
- Evidence: Local directory contains 94 train JPEGs, 10 validation JPEGs, and one COCO JSON per split.
- Trade-offs: Requires additional derived storage and explicit scripts.
- When to revisit: Only if a new source export supersedes this one.

## ADR-0002: Keep blue separate from binary health labels

- Date: 2026-09-14
- Status: Accepted
- Context: COCO metadata contains blue, green, and yellow categories. The project semantics define blue as locate/identify, not health.
- Decision: Map only green to class 0 and yellow to class 1 in later derived data; exclude blue from binary loss and report the missing blue handling as an open deployment issue.
- Why: Prevents label leakage and false health semantics.
- Alternatives considered: Relabel blue as green or unknown; rejected because it changes the stated task and hides an operational state.
- Evidence: Both COCO files explicitly list `blue`, `green`, and `yellow`.
- Trade-offs: The deployed binary classifier will need explicit abstention, an ROI rule, or a separate blue detector if blue appears in the field.
- When to revisit: After EDA and field-camera review.

## ADR-0003: Do not use the provided validation split as health validation without review

- Date: 2026-09-14
- Status: Accepted
- Context: The provided validation annotations contain 10 blue and 26 green objects but zero yellow objects.
- Decision: Phase 1 must evaluate leakage-safe regrouping or another documented split before model selection.
- Why: Fault recall cannot be measured on a validation set with no yellow examples.
- Alternatives considered: Proceed with the provided split; rejected because it cannot measure the primary fault class.
- Evidence: Direct COCO annotation count from both local JSON files.
- Trade-offs: A revised split may reduce comparability with the source export and may expose limited yellow data.
- When to revisit: After duplicate/session/scene analysis.

## ADR-0004: Treat exact-duplicate checks as insufficient for leakage exclusion

- Date: 2026-09-14
- Status: Accepted
- Context: Phase 1 found no exact SHA-256 duplicates, repeated source stems, or cross-split filenames. The export contains no reliable device/session grouping metadata.
- Decision: Phase 2 must document a leakage-aware split strategy and explicitly report that near-duplicate and device/session leakage cannot be ruled out from the current metadata alone.
- Why: Multiple views of the same physical LED can differ at the pixel level while still inflating evaluation results.
- Alternatives considered: Declare the provided train/valid split leakage-safe; rejected because exact hashes do not test scene, device, or session identity.
- Evidence: `data/interim/eda_report.json` and `docs/eda_report.md`.
- Trade-offs: A conservative split may reduce usable validation data, especially given only 10 yellow annotations.
- When to revisit: If source metadata or additional labeled captures become available.

## ADR-0005: Split derived crops by source image and exclude blue

- Date: 2026-09-14
- Status: Accepted
- Context: COCO frames can contain multiple labels, including green+yellow and blue+green+yellow. The provided validation split has no yellow annotations.
- Decision: Derive one padded 64x64 RGB crop per green/yellow annotation, exclude blue annotations, combine source directories, and assign every source image to exactly one deterministic train/valid/test split.
- Why: Keeps crops from the same frame together, creates validation/test yellow examples, and preserves the project’s binary semantics.
- Alternatives considered: Reuse the Roboflow train/valid directories; rejected because valid has zero yellow. Relabel blue as green; rejected because blue is locate/identify, not health.
- Evidence: `data/processed/led_crops/manifest.json` contains 339 crops: 329 green and 10 yellow, with zero source-image overlap across splits.
- Trade-offs: Only 1 yellow crop is in valid and 2 in test; device/session leakage remains unresolved without metadata.
- When to revisit: When more yellow data or device/session identifiers become available.

## ADR-0006: Use a small custom CNN as the first baseline

- Date: 2026-09-14
- Status: Accepted
- Context: The processed dataset has 339 crops but only 7 yellow training crops, 1 validation yellow, and 2 test yellows. The embedded target has tight resource constraints.
- Decision: Establish Candidate A with three convolution blocks, global average pooling, and a linear two-class head. Use weighted cross-entropy, deterministic training, and early stopping on validation loss.
- Why: This is the smallest model that tests the end-to-end pipeline while giving the rare fault class explicit loss weight.
- Alternatives considered: BatchNorm, Dropout, augmentation, and a pretrained backbone; deferred because their benefit and embedded cost are not measured yet.
- Evidence: `data/interim/baseline/metrics.json` reports 6,098 parameters, a 27,772-byte checkpoint, and test yellow recall 0/2.
- Trade-offs: The baseline is compact but currently fails to detect the two test yellow crops. The tiny validation/test fault counts make selection unstable.
- When to revisit: After more yellow data is collected or during Phase 4 architecture comparison with identical splits and metrics.

## ADR-0007: Defer transfer learning and model selection

- Date: 2026-09-14
- Status: Accepted
- Context: Candidate B is 1,056 parameters versus Candidate A at 6,098, but both have yellow test recall 0/2. `torchvision` and cached pretrained weights are unavailable.
- Decision: Measure Candidate B under the same protocol, defer Candidate C until a verified pretrained dependency and weights are available, and do not select a final model from the current two-yellow test set.
- Why: A higher accuracy here can result from predicting green for every fault; the embedded target also makes model size relevant, but not at the expense of missed faults.
- Alternatives considered: Select B on 96.7% accuracy; rejected because it misses both yellow tests. Download arbitrary pretrained weights; rejected because the evidence would not be reproducible or verified.
- Evidence: `data/interim/architecture_comparison/comparison.json` and `docs/architecture_comparison.md`.
- Trade-offs: The architecture phase produces no final winner and leaves transfer learning unmeasured, but preserves honest evidence and the stated fault-recall priority.
- When to revisit: After collecting more yellow examples and installing/verifying a compatible pretrained model source.

## ADR-0008: Calibrate static INT8 from the train split only

- Date: 2026-09-14
- Status: Accepted
- Context: Phase 5 compared float32, dynamic-range, full INT8, and QAT versions of Candidate B. The dataset has only 7 yellow train crops and 2 yellow test crops.
- Decision: Use all 228 train crops for static calibration, never validation or test crops, and report serialized PyTorch sizes separately from future TFLite/embedded sizes.
- Why: Prevents evaluation leakage and preserves a valid deployment comparison boundary.
- Alternatives considered: Calibrate with all processed crops; rejected because it leaks validation/test distributions. Select a mode by PyTorch state-dict size; rejected because packed quantized metadata makes that size incomparable to final TFLite output.
- Evidence: `data/interim/quantization/quantization_comparison.json` records 228 calibration samples and identical `0/2` yellow recall across modes.
- Trade-offs: Quantization did not improve the observed fault metric, and final deployment size remains unknown until Phase 6 conversion.
- When to revisit: During TFLite conversion and after collecting more yellow examples.

## ADR-0009: Block TFLite claims until the export toolchain and weight mapping are verified

- Date: 2026-09-15
- Status: Blocked
- Context: Phase 6 requires a TFLite artifact and desktop verification. TensorFlow, TFLite Runtime, ONNX, and ONNX-to-TFLite packages are absent from the configured environment, and installing TensorFlow failed.
- Decision: Do not claim TFLite conversion, deployment-format size, or desktop TFLite parity. Provide a guarded export entry point and resume only after a verified converter and an explicit PyTorch-to-Keras weight mapping are available.
- Why: A PyTorch quantized state dictionary is not itself a TFLite flatbuffer, and silently changing model weights would invalidate prior measurements.
- Alternatives considered: Treat the PyTorch checkpoint as TFLite; rejected because formats and operators differ. Download/install an unverified package outside the configured environment; rejected because the result would not be reproducible.
- Evidence: `src/export_tflite.py`, `docs/phase6_blocker.md`, and the failed workspace TensorFlow installation attempt.
- Trade-offs: Phase 6 is paused, but no unsupported deployment claim enters the project record.
- When to revisit: Once TensorFlow or another verified PyTorch-to-TFLite path is installed.

## ADR-0010: Export float32 TFLite through an explicitly verified Keras mapping

- Date: 2026-09-15
- Status: Accepted
- Context: TensorFlow CPU 2.21.0 became available in the workspace `.venv`. Candidate B is stored as a PyTorch checkpoint, while TFLite conversion operates on the Keras graph used for export.
- Decision: Extract PyTorch weights and reference logits, map them into an equivalent Keras depthwise-separable graph, convert float32 to TFLite, and require desktop parity on all 61 test crops.
- Why: This verifies the actual tensor layout and preprocessing contract instead of treating a PyTorch checkpoint as a TFLite model.
- Alternatives considered: Convert the PyTorch checkpoint directly; rejected because no direct converter was verified. Claim INT8 TFLite from the PyTorch INT8 artifact; rejected because the deployment formats and quantization metadata differ.
- Evidence: `data/interim/tflite/desktop_verification.json` reports max absolute logit errors below `2e-7` and a 9,848-byte flatbuffer.
- Trade-offs: The verified artifact is float32; TFLite INT8 conversion and embedded parity remain later work.
- When to revisit: During deployment conversion with train-only representative data.

## ADR-0011: Separate embedded artifact generation from hardware claims

- Date: 2026-09-15
- Status: Blocked
- Context: The TFLite flatbuffer and desktop parity fixture can be generated, but no ESP build tool or connected Robodo ESP32-CAM is available.
- Decision: Generate a TFLite Micro C byte array and 61-sample parity fixture, provide an integration skeleton, and block Phase 7 hardware completion until compilation and on-device measurements are performed.
- Why: A C array and desktop interpreter result do not establish TFLite Micro operator compatibility, tensor arena size, latency, or camera preprocessing parity.
- Alternatives considered: Mark the phase done from desktop parity alone; rejected because it would claim unmeasured embedded behavior. Treat the 64 KiB arena as final; rejected because it is a placeholder.
- Evidence: `embedded/bundle_metadata.json`, `embedded/tflite_micro_inference.cpp`, and `docs/phase7_blocker.md`.
- Trade-offs: Artifact integration is ready, but hardware metrics remain pending.
- When to revisit: When the ESP toolchain and Robodo ESP32-CAM are available.

## ADR-0012: Provide a desktop upload path while hardware integration is paused

- Date: 2026-09-16
- Status: Accepted
- Context: The user has paused physical ESP32 requirements but needs to try the
  verified model with uploaded images.
- Decision: Add a dependency-light Python CLI/GUI using the verified float32
  TFLite model. Require a close-up LED crop and reuse RGB, 64x64 Lanczos
  resize, and `[0, 1]` float32 preprocessing.
- Why: This enables immediate, reproducible testing without pretending that a
  classifier can find an LED in a full server photograph.
- Alternatives considered: Add object detection; rejected because it changes
  the model scope and requires new training/evaluation. Add a web framework;
  rejected because the existing environment already provides Python, TensorFlow,
  and Pillow without another runtime dependency.
- Evidence: `src/classify_image.py` and the documented CLI/GUI commands.
- Trade-offs: Users must crop the LED manually; blue LEDs can still be forced
  into the binary green/yellow output.
- When to revisit: If field images routinely contain multiple LEDs, full scenes,
  or blue locate indicators.

## ADR-0013: Treat detected blue as the unhealthy output in the desktop policy

- Date: 2026-09-16
- Status: Accepted
- Context: The user requires blue indicators to be handled as unhealthy, even
  though blue was excluded from binary model training.
- Decision: Detect sufficiently visible blue pixels after the model's 64x64 RGB
  preprocessing and override the prediction to class 1 (`yellow/fault`). Keep
  the raw model logits and expose `blue_detected` for auditability.
- Why: Makes the requested field policy explicit without relabeling blue in the
  training data or pretending the model learned a blue class.
- Alternatives considered: Retrain blue as yellow; rejected because it would
  change the training labels and invalidate prior measurements. Keep blue
  unknown; rejected because it does not meet the requested operational policy.
- Evidence: The classifier now reports both the raw model prediction and the
  policy result; focused inference on the supplied image is the verification.
- Trade-offs: Any sufficiently blue region in an input crop triggers unhealthy,
  so full scenes and blue non-LED objects can produce false faults. The
  threshold is a heuristic and needs field-camera calibration.
- When to revisit: When blue LED crops and negative examples are available for
  a measured detector or a three-class model.

## ADR-0014: Use a measured single-app PlatformIO layout for Phase 7

- Date: 2026-10-11
- Status: Accepted
- Context: The ESP32-CAM became available through a CH340 adapter. The float32
  TFLite artifact needed on-device compilation, parity, and resource
  measurements.
- Decision: Use PlatformIO `espressif32@6.12.0`, Arduino ESP32 framework
  `3.20017.241212`, and `sebastianregelmann/tflite-micro@1.26.41`, with a
  custom single-app 4 MB partition and a 256 KiB PSRAM tensor arena.
- Why: The default 3 MB application partition could not contain the linked
  TFLite Micro firmware; the measured board has 4 MB flash and PSRAM.
- Alternatives considered: Treat the 64 KiB skeleton arena as final; rejected
  because allocation and runtime measurements are required. Use the default
  partition; rejected because the 3,354,405-byte firmware exceeded 3,145,728
  bytes.
- Evidence: 61/61 parity success, 586,318 us mean invoke latency, successful
  OV2640 capture, and `docs/deployment.md`.
- Trade-offs: Float32 inference is slow at about 586 ms per invoke and the
  camera crop/resize is not yet training-pipeline equivalent.
- When to revisit: After representative camera captures are collected and
  INT8 conversion is measured.

## ADR-0015: Generate a full-integer TFLite deployment candidate

- Date: 2026-10-11
- Status: Accepted
- Context: Float32 inference is measured at about 586 ms on the ESP32-CAM.
  PyTorch quantization results do not establish TFLite Micro behavior.
- Decision: Export a separate full-integer TFLite model with int8 input and
  output, calibrated from all 228 training crops and no validation/test crops.
- Why: This creates a deployment artifact that can be measured against the
  float32 firmware without changing the verified float32 model.
- Alternatives considered: Flash the PyTorch INT8 checkpoint; rejected because
  it is not a TFLite Micro flatbuffer. Reuse the float32 model; rejected for
  the latency investigation because it has no quantized operators.
- Evidence: `data/interim/tflite/int8_desktop_verification.json` reports a
  9,720-byte flatbuffer and 0.00188455 maximum dequantized logit delta.
  ESP32 measurement reports 61 completed parity samples, maximum one raw
  output-count difference, and 592,041 us mean invoke latency.
- Trade-offs: INT8 reduces the firmware image from 3,424,640 to 1,105,744
  bytes, but did not reduce measured latency and still misses both yellow test
  crops. A one-count parity tolerance is required because of quantized-kernel
  rounding.
- When to revisit: After representative camera captures are collected and the
  INT8 camera preprocessing path is measured.

## ADR-0016: Reuse the measured camera preprocessing path for INT8

- Date: 2026-10-11
- Status: Accepted
- Context: The float32 firmware already captured a 160x120 RGB565 frame,
  selected a centered 120x120 crop, and resized it to 64x64 by nearest
  neighbor. The INT8 model requires the same RGB input contract followed by
  int8 quantization.
- Decision: Add an INT8 camera inference path using the same crop, RGB565
  channel conversion, nearest-neighbor resize, and input quantization
  (`scale=0.0039215689`, `zero_point=-128`). Dequantize the two output values
  using the measured output parameters for logging and select the larger logit.
- Why: This isolates the quantized model integration while keeping camera
  behavior comparable to the existing float32 demonstration.
- Alternatives considered: Rework camera preprocessing to match annotated
  training crops immediately; rejected because the camera has no annotation
  metadata and that policy requires representative capture review first.
- Evidence: COM8 run captured a 160x120 RGB565 frame and completed INT8
  inference in 592,946 microseconds.
- Trade-offs: The path is operationally complete but not evidence that the
  centered crop contains the LED or that nearest-neighbor preprocessing
  matches the training-time padded crop and Lanczos resize.
- When to revisit: After collecting representative camera images and defining
  the ROI/crop and blue-indicator policy.

## ADR-0017: Validate the camera tensor contract before changing the crop

- Date: 2026-10-12
- Status: Accepted
- Context: The ESP32 camera path runs successfully, but no captured frame
  fixture has yet been retained to compare the crop and resize policy with
  the training pipeline.
- Decision: Add a deterministic desktop contract check for the current
  160x120 RGB565 frame assumptions, centered 120x120 crop, floor-index nearest
  resize, RGB565 channel extraction, and int8 input quantization.
- Why: It catches coordinate, shape, channel-order, and quantization-range
  errors before collecting or interpreting representative LED images.
- Alternatives considered: Change the crop policy immediately; rejected
  because LED placement has not been measured. Claim camera/training parity
  from a successful inference; rejected because successful execution does not
  establish semantic preprocessing equivalence.
- Evidence: `src/validate_camera_preprocessing.py` passed and produced
  `data/interim/camera_preprocessing_validation.json`.
- Trade-offs: The check validates implementation mechanics only; it cannot
  determine whether the fixed crop contains the LED or whether Lanczos and
  nearest-neighbor preprocessing produce equivalent predictions.
- When to revisit: After obtaining representative camera captures with known
  LED locations and colors.

## ADR-0018: Export raw camera frames over serial for policy validation

- Date: 2026-10-12
- Status: Accepted
- Context: A successful model invocation cannot show whether the fixed crop
  contains the LED or whether the camera sees the expected color. The board
  has no configured storage or network capture service.
- Decision: Emit one framed 160x120 RGB565 image during INT8 firmware startup
  and decode it with `src/capture_camera_frame.py` into a PNG under
  `data/interim/camera_captures/`.
- Why: Serial export uses the existing FTDI connection and preserves the
  pixels used by the camera path without silently changing the inference
  preprocessing.
- Alternatives considered: Add Wi-Fi streaming; rejected because it adds
  unrelated networking complexity and changes the deployment surface. Use a
  screenshot of an external monitor; rejected because it is not the actual
  camera frame.
- Evidence: A 38,400-byte RGB565 frame was captured and decoded successfully
  as `data/interim/camera_captures/camera_frame_unlabeled.png`.
- Trade-offs: The current capture is unlabeled and dark, so it is a transport
  validation only. Known green, yellow, and blue scenes must be captured
  before selecting a crop or blue-indicator policy.
- When to revisit: After representative labeled captures are collected.

## ADR-0019: Quantify camera exposure before selecting an ROI

- Date: 2026-10-12
- Status: Accepted
- Context: The first decoded frame was too dark and had no known LED label.
  Selecting a crop or blue policy from that image would be speculation.
- Decision: Analyze captured frames for brightness, brightest-region bounds,
  color dominance, and fixed-crop coverage before changing the camera policy.
- Why: Separates camera exposure/transport problems from ROI placement problems
  using measurable evidence.
- Alternatives considered: Move the crop or add automatic detection immediately;
  rejected because neither can be justified without an adequately exposed,
  known-color capture.
- Evidence: `src/analyze_camera_capture.py` produces a JSON report and the
  existing unlabeled frame is retained as a transport-only artifact.
- Trade-offs: This analysis cannot identify green/yellow/blue semantics and
  requires user-labeled captures for a deployment decision.
- When to revisit: After one adequately exposed capture for each known LED
  color.

## ADR-0020: Use length-framed software JPEG for serial camera captures

- Date: 2026-10-12
- Status: Accepted
- Context: Raw 38,400-byte RGB565 dumps repeatedly decoded as nearly black,
  while the same sensor produced a valid live stream. The raw serial protocol
  had no explicit payload length and operated at 115200 baud.
- Decision: Convert the captured RGB565 frame to software JPEG on the ESP32,
  send a `camera_jpeg_begin:<length>` header and terminator at 921600 baud,
  and decode the JPEG directly on the host.
- Why: Length framing makes payload boundaries explicit and JPEG provides a
  standard image representation for visual and brightness checks.
- Alternatives considered: Continue parsing raw RGB565; rejected because the
  captured stream did not provide reliable evidence despite a working camera.
  Use hardware JPEG; rejected because this detected sensor reports
  `ESP_ERR_NOT_SUPPORTED` for JPEG pixel format.
- Evidence: `fixed_capture_20261012_155533.jpg` is 160x120 with mean
  brightness `247.14/255`; 82.1% of detected bright pixels fall inside the
  fixed crop.
- Trade-offs: Software JPEG consumes CPU and transient memory, and this is a
  diagnostic transfer path rather than the deployed inference path. The
  inference path still consumes RGB565 directly.
- When to revisit: When representative known-color captures are collected and
  the diagnostic exporter is no longer needed.

## ADR-0021: Classify LED policy directly from the standalone livestream

- Date: 2026-10-12
- Status: Accepted
- Context: The camera livestream is visually valid, while the earlier raw
  serial frame transport was unreliable. Blue is a locate/identify indicator,
  not one of the binary health labels. The trained green/yellow CNN cannot be
  run by the separate stream firmware without adding the model and TFLite
  Micro runtime to that diagnostic environment.
- Decision: Keep the livestream firmware separate from the CNN classifier and
  classify the currently displayed MJPEG frame in a browser canvas every
  500 ms using explicit green, yellow, and the existing blue policy. Return
  unknown when no policy reaches the minimum visible-pixel threshold.
- Why: This provides continuous live classification without contending for a
  second framebuffer capture, while avoiding a false claim that the browser
  result is TFLite inference or relabeling blue as green/yellow.
- Alternatives considered: Add a second `/blue` camera-capture endpoint;
  rejected initially because a single-frame capture could contend with the
  stream. Integrate blue as a learned health class; rejected because it
  violates the defined binary health semantics.
- Evidence: The browser page shows the live stream, reports visible color
  fractions, and updates the classification from the displayed image after
  flashing `esp32cam_stream`.
- Trade-offs: Browser-side analysis depends on the displayed stream and uses
  uncalibrated color heuristics. It is not TFLite inference and does not prove
  server health.
- When to revisit: After collecting labeled blue/non-blue captures under the
  intended camera and lighting conditions.

## ADR-0022: Classify the rover stream on the laptop

- Date: 2026-10-12
- Status: Accepted
- Context: The rover needs to carry the ESP32-CAM while the laptop receives
  the stream over Wi-Fi. The current ESP32 stream handler is a blocking
  single-client handler and the board has one camera framebuffer.
- Decision: Keep the ESP32 firmware responsible for camera capture and MJPEG
  transport. Add a Streamlit laptop application that consumes `/stream`,
  applies the same explicit color policy, and displays the frame, status, and
  color fractions. Operate the dashboard as the sole stream client.
- Why: This keeps image classification and UI resources on the laptop,
  avoids adding the TFLite runtime to the diagnostic stream firmware, and
  provides the rover integration requested over Wi-Fi.
- Alternatives considered: Run the CNN on every stream frame on the ESP32;
  rejected for now because the current stream firmware is separate and the
  validated CNN has poor yellow recall (`0/2` test faults). Support multiple
  ESP32 viewers; deferred because it requires a non-blocking/multiplexed
  stream design and is unnecessary for one rover dashboard.
- Evidence: The dashboard returned HTTP 200 locally, and the ESP32 endpoint
  returned valid multipart MJPEG/JPEG data when no other client held the
  stream.
- Trade-offs: The dashboard is color-policy based rather than TFLite CNN
  inference, and one direct stream viewer cannot run concurrently.
- When to revisit: After labeled green/yellow/blue camera captures and if
  multiple viewers or remote clients are required.

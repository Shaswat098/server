#include <Arduino.h>
#include <cmath>
#include <esp_camera.h>
#include "esp_heap_caps.h"

#include "../model_data.h"
#include "../parity_fixture.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr size_t kTensorArenaBytes = 256 * 1024;
constexpr size_t kSamplesToRun = kParitySampleCount;
uint8_t* tensor_arena = nullptr;

camera_config_t camera_config() {
  camera_config_t config{};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sccb_sda = 26;
  config.pin_sccb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QQVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  return config;
}

bool AddOperators(tflite::MicroMutableOpResolver<6>& resolver) {
  return resolver.AddDepthwiseConv2D() == kTfLiteOk &&
         resolver.AddConv2D() == kTfLiteOk &&
         resolver.AddRelu() == kTfLiteOk &&
         resolver.AddMaxPool2D() == kTfLiteOk &&
         resolver.AddMean() == kTfLiteOk &&
         resolver.AddFullyConnected() == kTfLiteOk;
}

float rgb565_channel(uint16_t pixel, uint16_t mask, uint8_t shift, float scale) {
  return static_cast<float>((pixel & mask) >> shift) / scale;
}

bool RunCameraInference(tflite::MicroInterpreter& interpreter) {
  camera_config_t config = camera_config();
  const esp_err_t camera_status = esp_camera_init(&config);
  Serial.printf("camera_init=%s\n", esp_err_to_name(camera_status));
  if (camera_status != ESP_OK) {
    return false;
  }
  camera_fb_t* frame = esp_camera_fb_get();
  if (frame == nullptr) {
    Serial.println("camera_capture=failed");
    return false;
  }
  Serial.printf("camera_capture=ok\nframe_width=%u\nframe_height=%u\nframe_length=%u\n",
                frame->width, frame->height, frame->len);
  TfLiteTensor* input = interpreter.input(0);
  constexpr uint16_t kCropSize = 120;
  constexpr uint16_t kCropLeft = 20;
  for (uint16_t y = 0; y < 64; ++y) {
    const uint16_t source_y = static_cast<uint16_t>(y * kCropSize / 64);
    for (uint16_t x = 0; x < 64; ++x) {
      const uint16_t source_x =
          static_cast<uint16_t>(kCropLeft + x * kCropSize / 64);
      const size_t offset =
          (static_cast<size_t>(source_y) * frame->width + source_x) * 2;
      const uint16_t pixel =
          (static_cast<uint16_t>(frame->buf[offset]) << 8) | frame->buf[offset + 1];
      const size_t output_index =
          (static_cast<size_t>(y) * 64 + x) * 3;
      input->data.f[output_index] =
          rgb565_channel(pixel, 0xF800, 11, 31.0f);
      input->data.f[output_index + 1] =
          rgb565_channel(pixel, 0x07E0, 5, 63.0f);
      input->data.f[output_index + 2] =
          rgb565_channel(pixel, 0x001F, 0, 31.0f);
    }
  }
  const uint32_t invoke_start = micros();
  const TfLiteStatus invoke_status = interpreter.Invoke();
  const uint32_t invoke_us = micros() - invoke_start;
  if (invoke_status != kTfLiteOk) {
    esp_camera_fb_return(frame);
    Serial.println("camera_inference=failed");
    return false;
  }
  TfLiteTensor* output = interpreter.output(0);
  const uint8_t predicted_class = output->data.f[0] > output->data.f[1] ? 0 : 1;
  Serial.printf("camera_inference=ok\ncamera_inference_us=%u\n"
                "camera_green_logit=%.6f\ncamera_yellow_logit=%.6f\n"
                "camera_class=%u\n",
                invoke_us, output->data.f[0], output->data.f[1],
                predicted_class);
  esp_camera_fb_return(frame);
  return true;
}

void print_measurements() {
  Serial.printf("chip_model=%s\n", ESP.getChipModel());
  Serial.printf("cpu_mhz=%u\n", ESP.getCpuFreqMHz());
  Serial.printf("flash_size=%u\n", ESP.getFlashChipSize());
  Serial.printf("free_heap=%u\n", ESP.getFreeHeap());
  Serial.printf("psram_found=%s\n", psramFound() ? "true" : "false");
  Serial.printf("free_psram=%u\n", ESP.getFreePsram());
}
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\nphase7_tflm_start");
  print_measurements();

  tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kTensorArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr) {
    Serial.println("tensor_arena=allocation_failed");
    return;
  }
  Serial.printf("tensor_arena_bytes=%u\n", kTensorArenaBytes);

  const tflite::Model* model = tflite::GetModel(kCandidateBModel);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("model_schema=%u expected=%u\n", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  tflite::MicroMutableOpResolver<6> resolver;
  if (!AddOperators(resolver)) {
    Serial.println("operator_registration=failed");
    return;
  }
  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaBytes);
  const uint32_t allocation_start = micros();
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("allocate_tensors=failed");
    return;
  }
  Serial.printf("allocate_tensors=ok\nallocation_us=%u\n", micros() - allocation_start);

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  Serial.printf("input_type=%d\noutput_type=%d\n", input->type, output->type);
  Serial.printf("free_heap_after_allocate=%u\n", ESP.getFreeHeap());
  Serial.printf("free_psram_after_allocate=%u\n", ESP.getFreePsram());

  bool parity_ok = true;
  uint32_t total_invoke_us = 0;
  for (size_t sample = 0; sample < kSamplesToRun; ++sample) {
    Serial.printf("parity_sample_start=%u\n", sample);
    for (size_t index = 0; index < kParityInputElements; ++index) {
      input->data.f[index] =
          kParityInputs[sample * kParityInputElements + index];
    }
    const uint32_t invoke_start = micros();
    if (interpreter.Invoke() != kTfLiteOk) {
      parity_ok = false;
      break;
    }
    total_invoke_us += micros() - invoke_start;
    Serial.printf("parity_sample_invoke_us=%u\n", micros() - invoke_start);
    for (size_t index = 0; index < kParityOutputElements; ++index) {
      const float expected =
          kParityExpectedLogits[sample * kParityOutputElements + index];
      if (fabsf(output->data.f[index] - expected) > 1e-4f) {
        parity_ok = false;
      }
    }
  }
  Serial.printf("parity_samples=%u\n", kSamplesToRun);
  Serial.printf("parity_ok=%s\n", parity_ok ? "true" : "false");
  Serial.printf("mean_invoke_us=%u\n", total_invoke_us / kSamplesToRun);
  RunCameraInference(interpreter);
  print_measurements();
  Serial.println("phase7_tflm_end");
}

void loop() {
  delay(5000);
}

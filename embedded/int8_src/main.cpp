#include <Arduino.h>
#include <cmath>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include "img_converters.h"

#include "../int8_bundle/model_data.h"
#include "../int8_bundle/parity_fixture.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr size_t kTensorArenaBytes = 256 * 1024;
constexpr int kParityOutputTolerance = 1;
constexpr float kInputScale = 0.003921568859368563f;
constexpr int kInputZeroPoint = -128;
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
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QQVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_DRAM;
  return config;
}

float rgb565_channel(uint16_t pixel, uint16_t mask, uint8_t shift, float scale) {
  return static_cast<float>((pixel & mask) >> shift) / scale;
}

bool run_camera_inference(tflite::MicroInterpreter& interpreter) {
  camera_config_t config = camera_config();
  const esp_err_t camera_status = esp_camera_init(&config);
  Serial.printf("camera_init=%s\n", esp_err_to_name(camera_status));
  if (camera_status != ESP_OK) {
    return false;
  }
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    sensor->set_brightness(sensor, 2);
    sensor->set_contrast(sensor, 1);
    sensor->set_saturation(sensor, 1);
    sensor->set_gain_ctrl(sensor, 1);
    sensor->set_exposure_ctrl(sensor, 1);
    sensor->set_awb_gain(sensor, 1);
    Serial.println("camera_sensor_controls=brightness_2_contrast_1_saturation_1_auto_gain_auto_exposure_auto_awb");
  } else {
    Serial.println("camera_sensor_controls=unavailable");
  }
  camera_fb_t* frame = esp_camera_fb_get();
  if (frame == nullptr) {
    Serial.println("camera_capture=failed");
    return false;
  }
  Serial.printf("camera_capture=ok\ncamera_preprocessing=center_crop_120_nearest_resize\n"
                "camera_input_quantization=scale_0.0039215689_zero_point_-128\n"
                "frame_width=%u\nframe_height=%u\n"
                "frame_length=%u\n",
                frame->width, frame->height, frame->len);
  uint8_t* jpeg_buffer = nullptr;
  size_t jpeg_length = 0;
  if (!frame2jpg(frame, 80, &jpeg_buffer, &jpeg_length)) {
    esp_camera_fb_return(frame);
    Serial.println("camera_jpeg_capture=failed");
    return false;
  }
  Serial.printf("camera_jpeg_begin:%u\n", jpeg_length);
  Serial.write(jpeg_buffer, jpeg_length);
  Serial.flush();
  Serial.println("\ncamera_jpeg_end");
  free(jpeg_buffer);

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
      const float red = rgb565_channel(pixel, 0xF800, 11, 31.0f);
      const float green = rgb565_channel(pixel, 0x07E0, 5, 63.0f);
      const float blue = rgb565_channel(pixel, 0x001F, 0, 31.0f);
      input->data.int8[output_index] = static_cast<int8_t>(
          constrain(lroundf(red / kInputScale + kInputZeroPoint), -128L, 127L));
      input->data.int8[output_index + 1] = static_cast<int8_t>(
          constrain(lroundf(green / kInputScale + kInputZeroPoint), -128L, 127L));
      input->data.int8[output_index + 2] = static_cast<int8_t>(
          constrain(lroundf(blue / kInputScale + kInputZeroPoint), -128L, 127L));
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
  const TfLiteTensor* output = interpreter.output(0);
  const float green_logit =
      (static_cast<float>(output->data.int8[0]) + 39.0f) * 0.0025411166716367006f;
  const float yellow_logit =
      (static_cast<float>(output->data.int8[1]) + 39.0f) * 0.0025411166716367006f;
  const uint8_t predicted_class = green_logit > yellow_logit ? 0 : 1;
  Serial.printf("camera_inference=ok\ncamera_inference_us=%u\n"
                "camera_green_logit=%.6f\ncamera_yellow_logit=%.6f\n"
                "camera_class=%u\n",
                invoke_us, green_logit, yellow_logit, predicted_class);
  esp_camera_fb_return(frame);
  return true;
}

bool add_operators(tflite::MicroMutableOpResolver<6>& resolver) {
  return resolver.AddDepthwiseConv2D() == kTfLiteOk &&
         resolver.AddConv2D() == kTfLiteOk &&
         resolver.AddRelu() == kTfLiteOk &&
         resolver.AddMaxPool2D() == kTfLiteOk &&
         resolver.AddMean() == kTfLiteOk &&
         resolver.AddFullyConnected() == kTfLiteOk;
}
}  // namespace

void setup() {
  Serial.begin(921600);
  delay(1000);
  Serial.println("\nphase7_int8_start");
  Serial.printf("chip_model=%s\ncpu_mhz=%u\nflash_size=%u\nfree_heap=%u\n"
                "psram_found=%s\nfree_psram=%u\n",
                ESP.getChipModel(), ESP.getCpuFreqMHz(), ESP.getFlashChipSize(),
                ESP.getFreeHeap(), psramFound() ? "true" : "false",
                ESP.getFreePsram());

  tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kTensorArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr) {
    Serial.println("tensor_arena=allocation_failed");
    return;
  }
  Serial.printf("tensor_arena_bytes=%u\n", kTensorArenaBytes);

  const tflite::Model* model = tflite::GetModel(kCandidateBModel);
  tflite::MicroMutableOpResolver<6> resolver;
  if (model->version() != TFLITE_SCHEMA_VERSION || !add_operators(resolver)) {
    Serial.println("model_or_operators=failed");
    return;
  }
  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("allocate_tensors=failed");
    return;
  }
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  Serial.printf("allocate_tensors=ok\ninput_type=%d\noutput_type=%d\n"
                "free_heap_after_allocate=%u\nfree_psram_after_allocate=%u\n",
                input->type, output->type, ESP.getFreeHeap(), ESP.getFreePsram());

  bool parity_ok = input->type == kTfLiteInt8 && output->type == kTfLiteInt8;
  uint32_t total_invoke_us = 0;
  size_t mismatch_count = 0;
  int max_abs_diff = 0;
  bool first_mismatch_reported = false;
  for (size_t sample = 0; sample < kInt8ParitySampleCount && parity_ok; ++sample) {
    for (size_t index = 0; index < kInt8ParityInputElements; ++index) {
      input->data.int8[index] =
          kInt8ParityInputs[sample * kInt8ParityInputElements + index];
    }
    const uint32_t start = micros();
    if (interpreter.Invoke() != kTfLiteOk) {
      parity_ok = false;
      break;
    }
    total_invoke_us += micros() - start;
    for (size_t index = 0; index < kInt8ParityOutputElements; ++index) {
      const int actual = output->data.int8[index];
      const int expected =
          kInt8ParityExpectedOutputs[sample * kInt8ParityOutputElements + index];
      const int absolute_difference = abs(actual - expected);
      if (absolute_difference > max_abs_diff) {
        max_abs_diff = absolute_difference;
      }
      if (actual != expected) {
        ++mismatch_count;
        if (!first_mismatch_reported) {
          Serial.printf("first_mismatch_sample=%u\nfirst_mismatch_index=%u\n"
                        "first_mismatch_actual=%d\nfirst_mismatch_expected=%d\n",
                        sample, index, actual, expected);
          first_mismatch_reported = true;
        }
      }
    }
  }
  parity_ok = max_abs_diff <= kParityOutputTolerance;
  Serial.printf("parity_samples=%u\nparity_ok=%s\nmismatch_count=%u\n"
                "max_abs_output_diff=%d\nparity_tolerance=%d\n"
                "mean_invoke_us=%u\n",
                kInt8ParitySampleCount, parity_ok ? "true" : "false",
                mismatch_count, max_abs_diff, kParityOutputTolerance,
                total_invoke_us / kInt8ParitySampleCount);
  run_camera_inference(interpreter);
  Serial.printf("free_heap_end=%u\nfree_psram_end=%u\nphase7_int8_end\n",
                ESP.getFreeHeap(), ESP.getFreePsram());
}

void loop() {
  delay(5000);
}

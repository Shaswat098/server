#include "model_data.h"
#include "parity_fixture.h"

#include <cmath>

// Integrate this file with the project's pinned TFLite Micro checkout and
// provide a tensor arena sized from on-device measurement.
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr size_t kTensorArenaBytes = 64 * 1024;
uint8_t tensor_arena[kTensorArenaBytes];
}

bool RunCandidateBParityTest(tflite::MicroInterpreter* interpreter) {
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);
  if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
    return false;
  }
  for (size_t sample = 0; sample < kParitySampleCount; ++sample) {
    for (size_t index = 0; index < kParityInputElements; ++index) {
      input->data.f[index] = kParityInputs[sample * kParityInputElements + index];
    }
    if (interpreter->Invoke() != kTfLiteOk) {
      return false;
    }
    for (size_t index = 0; index < kParityOutputElements; ++index) {
      const float expected = kParityExpectedLogits[sample * kParityOutputElements + index];
      if (fabsf(output->data.f[index] - expected) > 1e-4f) {
        return false;
      }
    }
  }
  return true;
}
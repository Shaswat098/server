# Phase 3 Baseline Report

## Candidate A

The baseline is a small custom CNN:

`Conv(3,8) -> ReLU -> MaxPool -> Conv(8,16) -> ReLU -> MaxPool -> Conv(16,32) -> ReLU -> global average pooling -> Linear(32,2)`

BatchNorm and Dropout were not added because this is the first measured baseline and the dataset is very small. The model has 6,098 trainable parameters and the saved PyTorch checkpoint is 27,772 bytes.

## Training

- Input: processed RGB crops at 64x64
- Seed: 20260914
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Requested epochs: 60
- Completed epochs: 54
- Early stopping patience: 12
- Loss: weighted cross-entropy
- Train class counts: green 221, yellow 7
- Class weights: green 0.5158, yellow 16.2857
- Training time: 14.35 seconds on the available CPU

Weighted loss was used because missed yellow faults are more costly than false alarms and the train split contains only seven yellow crops. This weighting does not create information missing from the dataset.

## Results

The best validation-loss checkpoint was epoch 42:

- Validation: 42/50 correct, accuracy 0.84
- Validation yellow recall: 1/1 = 1.00
- Validation confusion matrix: `[[41, 8], [0, 1]]`

Final checkpoint test results:

- Test samples: 61
- Green samples: 59
- Yellow samples: 2
- Accuracy: 0.8852
- Green recall: 0.9153
- Yellow precision: 0.0
- Yellow recall: 0.0
- Yellow F1: 0.0
- Test confusion matrix: `[[54, 5], [2, 0]]`

## Interpretation

Accuracy is misleading here: the test set contains only two yellow crops, and the model missed both. The baseline demonstrates that the training path, export, and metrics work, but it does not establish reliable fault detection. The yellow sample count and unknown device/session grouping are the controlling limitations.

The full epoch history and machine-readable metrics are in `data/interim/baseline/metrics.json`; the model checkpoint is `data/interim/baseline/baseline_cnn.pt`.

LED color remains a proxy for server status, not proof of actual server health.

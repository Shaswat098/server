# Phase 4 Architecture Comparison

## Controlled setup

Candidate A and Candidate B used the same processed manifest, source-image grouped splits, seed `20260914`, RGB 64x64 inputs, weighted cross-entropy, Adam at learning rate `0.001`, batch size `32`, 60-epoch limit, and 12-epoch validation-loss patience.

Candidate C was not run. `torchvision` is not installed and no cached pretrained weights were available. No unverified download was used, so there is no measured transfer-learning result to report.

## Results

| Candidate | Parameters | Checkpoint bytes | Time (s) | Test accuracy | Yellow recall | Green recall |
|---|---:|---:|---:|---:|---:|---:|
| A custom CNN | 6,098 | 27,772 | 15.26 | 0.8852 | 0/2 = 0.0 | 54/59 = 0.9153 |
| B depthwise-separable | 1,056 | 9,832 | 18.86 | 0.9672 | 0/2 = 0.0 | 59/59 = 1.0 |
| C transfer learning | Not run | Not run | Not run | Not run | Not run | Not run |

Candidate A test confusion matrix: `[[54, 5], [2, 0]]`.

Candidate B test confusion matrix: `[[59, 0], [2, 0]]`.

Candidate B reduces parameter count by 82.7% and checkpoint size by 64.6% versus A, but its apparent accuracy advantage is the result of predicting green for every test yellow. Neither model detected either of the two yellow test crops. Therefore no accuracy-based winner is declared.

## Selection

Selection is deferred. The test set has only two yellow crops, so fault recall has no useful stability. Candidate B is the embedded-size hypothesis to carry into later experiments, but it is not validated as the safer health classifier. Additional yellow data and device/session metadata are needed before model selection.

LED color is a proxy for server status, not proof of actual server health.

## Reproduction

```powershell
& 'C:/Users/shash/AppData/Local/Programs/Python/Python311/python.exe' src/compare_architectures.py
```

Machine-readable evidence is in `data/interim/architecture_comparison/comparison.json`; Candidate B weights are in `data/interim/architecture_comparison/candidate_b_depthwise.pt`.

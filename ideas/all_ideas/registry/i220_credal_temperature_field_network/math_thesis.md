# Math Thesis

## Working thesis

Near-puzzle ambiguity is best handled as sample-wise calibration of an
ordinary binary classifier rather than as a Dirichlet-evidence head. The
encoder predicts a positive temperature `T(x)` and a bounded smoothing
factor `alpha(x)` from current-board fields; the classification logit is

```text
T(x)     = softplus(t(x)) + eps
alpha(x) = max_alpha * sigmoid(a(x))
logit(x) = (1 - alpha(x)) * z(x) / T(x).
```

## Claim

Sample-wise temperature should improve calibration on fine-label-1
(near-puzzle) examples and reduce overconfident false positives without
degrading AUROC.

## Falsifiers

- `fixed_temperature`: replace `T(x)` with a global learnable scalar.
- `ordinary_bce`: drop the calibration branch entirely.
- `temperature_stopgrad`: stop calibration gradients from shaping the
  encoder.

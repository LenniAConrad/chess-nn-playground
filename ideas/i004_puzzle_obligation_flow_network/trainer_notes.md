# Trainer Notes

## Shared Trainer Usage

Use the existing trainer:

```text
mode: puzzle_binary
num_classes: 1
loss: BCEWithLogitsLoss
```

## Special Losses

First implementation should use no auxiliary label loss.

Optional regularizers after baseline:

```text
allocation_entropy_weight: 0.001
residual_sparsity_weight: 0.001
```

Do not train with engine best moves or Stockfish scores.

## Logging Behavior

Log these additional diagnostics:

- mean residual by source class
- mean residual by obligation type
- allocation entropy
- max residual
- top dual price
- near-puzzle false-positive rate

## Metrics

Primary:

- test PR AUC
- test F1
- near-puzzle -> puzzle false-positive rate

Secondary:

- puzzle recall
- calibration
- random false-positive rate
- runtime per epoch


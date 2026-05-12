# Trainer Notes

## Shared Trainer Usage

Use the normal trainer with:

```text
mode: puzzle_binary
num_classes: 1
loss: BCEWithLogitsLoss
```

## Special Losses

First version should use no auxiliary loss. The minimax bottleneck itself is the experiment.

Optional later:

```text
reply_entropy_logging_only: true
```

Avoid using engine-selected best moves as supervision.

## Logging Behavior

Log:

- mean action count
- mean reply count
- reply entropy
- top action-response gap
- near-puzzle false-positive rate

## Metrics

Primary:

- test PR AUC
- test F1
- near-puzzle false-positive rate

Secondary:

- runtime per epoch
- recall at fixed near-puzzle FP thresholds


# Trainer Notes

## Shared Trainer Usage

Use:

```text
mode: puzzle_binary
num_classes: 1
loss: BCEWithLogitsLoss
```

## Special Losses

First implementation:

```text
main BCE only
```

Optional later:

```text
positive_null_margin_weight: 0.05
```

Only apply this to verified positives. Do not create labels for null positions.

## Logging Behavior

Log:

- current evidence
- null evidence
- contrast delta
- mean delta by source class
- near-puzzle false-positive rate

## Metrics

Primary:

- test PR AUC
- test F1
- near-puzzle false-positive rate

Secondary:

- puzzle recall
- calibration
- contrast separation between source classes


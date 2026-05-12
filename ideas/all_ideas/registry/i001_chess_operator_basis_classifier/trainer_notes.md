# Trainer Notes

## Shared Trainer Usage

Use the existing shared trainer:

```text
mode: puzzle_binary
loss: BCEWithLogitsLoss
num_classes: 1
```

## Special Losses

No special loss in the first version. Keep it architecture-only.

Optional later regularizer:

```text
relation_gate_entropy_weight: 0.001
```

to prevent all relation families from collapsing to one operator.

## Logging Behavior

In addition to existing metrics, log:

- mean relation gate per operator family
- operator-family ablation results
- near-puzzle false-positive rate

## Metrics

Primary:

- test PR AUC
- test F1
- near-puzzle -> puzzle false-positive rate

Secondary:

- random false-positive rate
- puzzle recall
- calibration curve


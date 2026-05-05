# Trainer Notes

## Shared Trainer Usage

Use:

```text
mode: puzzle_binary
num_classes: 1
loss: BCEWithLogitsLoss
```

## Special Losses

First version:

```text
main BCE only
```

Optional later:

```text
proof_gap_margin_weight: 0.05
tree_entropy_regularizer: 0.001
```

Do not supervise moves with engine best moves.

## Logging Behavior

Log:

- root proof cost
- root disproof cost
- proof-disproof gap
- selected proof line descriptors
- depth usage
- node counts
- near-puzzle false-positive rate

## Metrics

Primary:

- test PR AUC
- test F1
- near-puzzle -> puzzle false-positive rate

Benchmark target for this idea:

```text
test PR AUC >= 0.82
test F1 >= 0.76
near-puzzle FP <= 0.20
puzzle recall >= 0.78
```


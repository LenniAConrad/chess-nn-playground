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
main BCE
edit_sparsity_weight: 0.001
edit_entropy_weight: 0.001
```

Optional diagnostic regularizer after baseline:

```text
source_ordering_regularizer: off by default
```

Do not use source labels in the main training objective unless explicitly running a diagnostic experiment. The benchmark model should remain binary.

## Logging Behavior

Log:

- `E_plus`
- `E_minus`
- `edit_gap`
- top edit families for plus/minus
- mean energies by source class
- near-puzzle false-positive rate

## Metrics

Primary:

- test PR AUC
- test F1
- near-puzzle false-positive rate

Ambitious target:

```text
test PR AUC >= 0.82
test F1 >= 0.76
near-puzzle FP <= 0.20
puzzle recall >= 0.78
```


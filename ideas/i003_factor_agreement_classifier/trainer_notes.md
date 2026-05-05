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

The agreement term is architectural. Optional later:

```text
branch_dropout_weight
factor_diversity_weight
```

## Logging Behavior

Log:

- factor logits per sample
- disagreement value
- uncertainty value
- disagreement grouped by source class
- disagreement grouped by correct/incorrect prediction

## Metrics

Primary:

- test PR AUC
- test F1
- near-puzzle false-positive rate

Secondary:

- calibration
- recall at near-puzzle FP thresholds
- factor ablation table


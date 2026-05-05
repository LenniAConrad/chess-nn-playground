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
main BCE
selection_entropy_weight: small optional value
residual_bound: architectural, not a loss
```

Optional later:

```text
positive_deletion_gap_weight: 0.02
```

Only use verified puzzle labels for positive deletion-gap pressure.

## Logging Behavior

Log:

- selected token types
- selection entropy
- proof logit
- residual logit
- deletion gap
- near-puzzle false-positive rate

## Metrics

Primary:

- test PR AUC
- test F1
- near-puzzle false-positive rate

Secondary:

- puzzle recall
- deletion gap by source class
- witness type distribution


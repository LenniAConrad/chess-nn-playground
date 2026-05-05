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

Optional regularizers after a baseline:

```text
attacker_entropy_weight: 0.001
defender_entropy_weight: 0.001
payoff_l2_weight: 0.0001
```

Do not use engine best moves or Stockfish scores.

## Logging Behavior

Log:

- equilibrium value
- attacker entropy
- defender entropy
- exploitability
- top attacker/defender candidate types
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


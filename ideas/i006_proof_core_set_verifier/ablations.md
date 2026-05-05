# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `random_witnesses` | Select random candidate tokens. |
| `full_board_verifier` | Let verifier see all tokens. |
| `no_relations` | Remove deterministic relation features. |
| `unbounded_residual` | Let global residual bypass proof core. |
| `k_4_8_12_16` | Sweep witness budget. |
| `no_deletion_diagnostic` | Remove deletion pass from reports only. |

## What Each Ablation Tests

- `random_witnesses`: tests whether learned selection matters.
- `full_board_verifier`: tests whether sparse bottleneck helps hard negatives.
- `no_relations`: tests whether chess relations are necessary.
- `unbounded_residual`: tests whether global shortcuts dominate.
- `k_4_8_12_16`: tests proof-core size.

## Falsification Criteria

Reject if:

```text
random_witnesses matches learned witnesses
or full_board_verifier has equal near-puzzle FP and better PR AUC
or deletion_gap does not distinguish puzzles from near-puzzles
```

Reject the bottleneck if unbounded residual carries most of the performance.


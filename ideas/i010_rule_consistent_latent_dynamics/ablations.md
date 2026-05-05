# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `puzzle_only` | Disable all dynamics losses. |
| `legal_only_aux` | Use only legal/invalid move prediction. |
| `next_latent_only_aux` | Use only next-latent consistency. |
| `reconstruct_only_aux` | Use only next-board reconstruction. |
| `random_move_descriptors` | Destroy legal move semantics. |
| `easy_invalids_only` | Test whether invalid negatives are too easy. |
| `no_dynamics_summary_in_head` | Aux train encoder but final head sees only board latent. |

## What Each Ablation Tests

- `puzzle_only`: base trunk comparison.
- `legal_only_aux`: value of legality.
- `next_latent_only_aux`: value of consequence prediction.
- `reconstruct_only_aux`: value of board reconstruction.
- `random_move_descriptors`: chess rule semantics.
- `easy_invalids_only`: negative sampling quality.
- `no_dynamics_summary_in_head`: pretraining/regularization versus explicit dynamics features.

## Falsification Criteria

Reject if:

```text
puzzle_only matches all dynamics variants
or random_move_descriptors match legal descriptors
or high auxiliary accuracy gives no puzzle metric improvement
```

Also reject if runtime overhead is large and near-puzzle FP does not improve.


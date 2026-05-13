# Ablations

| ID | `ablation` | Effect | Falsifies |
|---|---|---|---|
| A0 | `none` | Full operator | Baseline |
| A1 | `shuffle_active_tokens` | In-batch permutation of active tokens | Primary falsifier -- whether the active set carries usable signal. |
| A2 | `diagonal_only` | Zero off-diagonal of `A` before solve | Whether the inverse-precision off-diagonals are load-bearing. If `A2` matches `none`, the operator collapses to per-channel rescaling. |
| A3 | `uniform_queries` | `Q = ones / r` | Whether trunk-conditioned query routing matters. |
| A4 | `zero_delta` | `primitive_delta = 0` | Recovers i193. |
| A5 | `trunk_only` | Alias of `zero_delta`. | |

## Decision rule

Keep p038 only if:

- The unablated run improves PR AUC or near-puzzle FP rate on the
  declared slice (high leverage-variance positions, i.e. one piece
  carrying most of the evidence) versus i193 baseline at matched recall.
- `shuffle_active_tokens` loses >=70% of the slice lift.
- `diagonal_only` loses >=40% of the slice lift.
- Aggregate PR AUC delta from i193 >= -0.005.
- Throughput drop versus i193 < 30%.

If any of these fails, drop p038.

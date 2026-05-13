# Ablations

| ID | `ablation` | Effect | Falsifies |
|---|---|---|---|
| A0 | `none` | Full operator | Baseline |
| A1 | `shuffle_grades_high` | In-batch permutation of `M^{(k)}` for `k >= 2` | Primary falsifier -- whether the higher-grade wedge components carry signal. |
| A2 | `first_order_only` | Zero `M^{(>=2)}` (degenerates to a sum pool) | Whether wedge cancellation is load-bearing. |
| A3 | `zero_delta` | `primitive_delta = 0` | Recovers i193. |
| A4 | `trunk_only` | Alias of `zero_delta` | |

## Decision rule

Keep p041 only if:

- The unablated run improves PR AUC or near-puzzle FP rate on the
  declared slice (high grade-2 magnitude positions) versus i193.
- `shuffle_grades_high` loses >=70% of the slice lift.
- `first_order_only` loses >=40% of the slice lift.
- Aggregate PR AUC delta from i193 >= -0.005.
- Throughput drop versus i193 < 30%.

If any of these fails, drop p041.

## Deferred extensions

- `D1`: increase `r` from 4 to 6 (cost goes from 15 to 42 output
  dimensions). Useful if grade-2 magnitude saturates at the small
  default.
- `D2`: anchor the wedge basis to chess-meaningful directions (king
  vector, central-control vector, file-pressure vector) instead of
  the learned `W` projection. Would test whether the antisymmetric
  cancellation needs domain priors.
- `D3`: cross-grade interactions -- emit `M^{(k)} * conj(M^{(k')})`
  pairs alongside the per-grade vectors. Would test whether the
  grade-decomposed representation alone is enough.

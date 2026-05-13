# Ablations

| ID | `ablation` | Effect | Falsifies |
|---|---|---|---|
| A0 | `none` | Full operator | Baseline |
| A1 | `shuffle_residual` | In-batch permutation of `Y` | Primary falsifier -- whether the residual carries usable signal. |
| A2 | `no_projection` | Set `M = 0`, recover plain weighted normalisation | Whether the conservation projection is load-bearing. |
| A3 | `uniform_weights` | Drop per-square weights (`D = I`) | Whether the rule-derived weight projection is load-bearing. |
| A4 | `zero_delta` | `primitive_delta = 0` | Recovers i193. |
| A5 | `trunk_only` | Alias of `zero_delta` | |

## Decision rule

Keep p040 only if:

- The unablated run improves PR AUC or near-puzzle FP rate on the
  declared slice (high explained-fraction positions) versus i193.
- `shuffle_residual` loses >=70% of the slice lift.
- `no_projection` loses >=40% of the slice lift (otherwise the
  projection step is not load-bearing).
- Aggregate PR AUC delta from i193 >= -0.005.
- Throughput drop versus i193 < 25%.

If any of these fails, drop p040.

## Deferred extensions

- `D1`: per-position material counts as additional charge columns.
- `D2`: learnable charge columns (mixing fixed-rule charges with
  trainable directions).
- `D3`: hierarchical version with multiple charge bases at different
  scales (per-square, per-file, per-region).

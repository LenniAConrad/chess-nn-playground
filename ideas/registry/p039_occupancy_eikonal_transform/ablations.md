# Ablations

| ID | `ablation` | Effect | Falsifies |
|---|---|---|---|
| A0 | `none` | Full operator | Baseline |
| A1 | `shuffle_field` | In-batch permutation of arrival field | Primary falsifier -- whether the field carries usable signal. |
| A2 | `single_iteration` | Cap relaxation at 1 step | Whether propagation beyond the immediate king-neighbour ring matters. |
| A3 | `uniform_costs` | All `c_uv = cost_bias` | Whether cost projection is load-bearing. |
| A4 | `zero_delta` | `primitive_delta = 0` | Recovers i193. |
| A5 | `trunk_only` | Alias of `zero_delta` | |

## Decision rule

Keep p039 only if:

- The unablated run improves PR AUC or near-puzzle FP rate on
  king-safety / escape-corridor positions versus i193 baseline.
- `shuffle_field` loses >=70% of the slice lift.
- `single_iteration` loses at least 30% of the slice lift (otherwise
  global propagation is not useful).
- Aggregate PR AUC delta from i193 >= -0.005.
- Throughput drop versus i193 < 25%.

If any of these fails, drop p039.

## Deferred extensions

- `D1`: per-edge anisotropic costs -- separate cost field per
  outgoing direction (8x more parameters).
- `D2`: rook / bishop / knight neighbour graphs in addition to king-
  neighbours -- multi-piece arrival fields.
- `D3`: temperature schedule -- start at high `tau`, anneal down so the
  field sharpens as training proceeds.

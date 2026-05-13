# Ablations

The model exposes five named ablations via the `model.ablation` config
field. The primary falsifier is `shuffle_logpartition`.

| ID | `ablation` | Effect on `forward()` | What it falsifies |
|---|---|---|---|
| A0 | `none` | Full operator | Baseline (unablated) |
| A1 | `shuffle_logpartition` | In-batch permutation of `y` | Whether the log-partition carries usable signal. Primary falsifier. |
| A2 | `uniform_edges` | `c_h`, `c_v` replaced with all-ones | Whether edge costs are load-bearing. |
| A3 | `uniform_sources` | `s`, `t` replaced with all-ones | Whether source/sink penalties are load-bearing. |
| A4 | `zero_delta` | `primitive_delta = 0` | Recovers i193 base logit. |
| A5 | `trunk_only` | Alias of `zero_delta` | Strongest baseline control. |

## Decision rule

Keep p037 only if:

- The unablated run improves PR AUC or near-puzzle FP rate on the
  declared slice (king-safety / fortress positions) versus i193 baseline.
- `shuffle_logpartition` loses >=70% of that slice lift.
- `uniform_edges` loses at least 40% of that lift (the edge structure
  has to matter, not just the source/sink budget).
- Aggregate PR AUC is no worse than 1.0% below i193 baseline.

If any of these fails, document the decision and drop p037.

## Deferred extensions

- `D1`: increase grid size to 6x6 (`2^W = 64` states) if the 4x4 grid
  exposes a slice lift but is bottlenecked by grid resolution.
- `D2`: anisotropic edge costs (different edge dimension per axis) --
  may help when chess king-safety prefers vertical / horizontal escape
  corridors over diagonal ones.
- `D3`: soft-temperature schedule -- start at high `tau`, anneal down so
  the cut becomes sharper as training proceeds.

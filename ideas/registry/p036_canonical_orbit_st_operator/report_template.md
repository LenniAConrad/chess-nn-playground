# Report Template

## Run

- Result path:
- Config:
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Architecture-Specific Diagnostics

- Mechanism family: `response_constraint`
- Primitive: Canonical-Orbit Straight-Through Operator (COSTO)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positive samples
  - Negative near-puzzle samples
- `primitive_delta` distribution on the same two buckets
- `cost_chosen_orbit_index` distribution (which of {e,F,R,FR} wins)
- `cost_orbit_gap` mean / median / quartiles
- `cost_orbit_ties` distribution (true ties have value >= 2)
- `cost_residual_norm` distribution

## Slice Findings

- Declared target slice: positions with `cost_orbit_gap` above median
  (positions that have a strong preferred orientation)
  - Required: p036 unablated >= i193 + 0.03 PR AUC on slice
  - Required: A1 (`shuffle_canonical`) loses >= 70% of that lift
- Watch slice: starting position symmetry / file-mirror duplicates
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean on positives | gate mean on negatives |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_canonical` | | | | |
| `identity_only` | | | | |
| `fixed_choice` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.03
- [ ] A1 (`shuffle_canonical`) loses >= 70% of the slice lift
- [ ] A2 (`identity_only`) loses >= 50% of the slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p036.

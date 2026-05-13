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

- Mechanism family: `king_path`
- Primitive: Blocker-Reset Ray Scan (BRRS)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positions with at least one slider
  - Positions without sliders
- `primitive_delta` distribution on the same two buckets
- `brrs_occupancy_density` vs label correlation
- `brrs_ray_magnitude` distribution

## Slice Findings

- Declared target slice: sliding-piece-dependent (pin / skewer /
  discovered attack / rook-on-open-file / queen line into king zone)
  - Required: p020 unablated >= i193 + 0.04 PR AUC on slice
  - Required: A1 (`zero_blocker`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean on sliders | gate mean on non-sliders |
|---|---|---|---|---|
| `none` | | | | |
| `zero_blocker` | | | | |
| `uniform_blocker` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.04
- [ ] A1 (`zero_blocker`) loses >= 70% of the slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p020.

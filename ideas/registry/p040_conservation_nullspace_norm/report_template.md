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
- Primitive: Conservation-Nullspace Normalization (CNN)
- `primitive_gate` mean / max / fraction > 0.5 on positives / negatives
- `primitive_delta` distribution
- `cnnorm_explained_frac` distribution
- `cnnorm_residual_norm` distribution

## Slice Findings

- Declared target slice: positions with `cnnorm_explained_frac` above
  median (positions where conservation projection explains most of the
  latent)
  - Required: p040 unablated >= i193 + 0.03 PR AUC on slice
  - Required: A1 (`shuffle_residual`) loses >= 70% of that lift
  - Required: A2 (`no_projection`) loses >= 40% of that lift
- Watch slice: positions with low explained fraction (the projection
  is empty so the operator should have low impact)
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean | explained frac |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_residual` | | | | |
| `no_projection` | | | | |
| `uniform_weights` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.03
- [ ] A1 (`shuffle_residual`) loses >= 70% of the slice lift
- [ ] A2 (`no_projection`) loses >= 40% of the slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p040.

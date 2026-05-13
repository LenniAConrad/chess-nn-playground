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
- Primitive: Woodbury Set Resolver (WSR)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positive samples
  - Negative near-puzzle samples
- `primitive_delta` distribution on the same two buckets
- `wsr_logdet_A` mean per label
- `wsr_leverage_mean` and `wsr_leverage_max` distributions

## Slice Findings

- Declared target slice: positions with leverage-variance above median
  (one piece carrying most of the evidence)
  - Required: p038 unablated >= i193 + 0.03 PR AUC on slice
  - Required: A1 (`shuffle_active_tokens`) loses >= 70% of that lift
  - Required: A2 (`diagonal_only`) loses >= 40% of that lift
- Watch slice: positions with many same-type pieces (correlated tokens)
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean on positives | gate mean on negatives |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_active_tokens` | | | | |
| `diagonal_only` | | | | |
| `uniform_queries` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.03
- [ ] A1 (`shuffle_active_tokens`) loses >= 70% of the slice lift
- [ ] A2 (`diagonal_only`) loses >= 40% of the slice lift
- [ ] Throughput drop versus i193 < 30%

If any box fails: drop p038.

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
- Primitive: Event-Symmetric Interaction Accumulator (ESIA)
- `primitive_gate` mean / max / fraction > 0.5
- `esia_order_<r>_magnitude` distribution for r = 1..R
- Correlation: `esia_order_2_magnitude` vs label

## Slice Findings

- Declared target slice: higher-order interaction tactics (fork,
  double-attack, discovered-attack triple, knight fork)
  - Required: p024 unablated >= i193 + 0.04 PR AUC
  - Required: A1 (`first_order_only`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | mean E^{(2)} magnitude | gate mean |
|---|---|---|---|---|
| `none` | | | | |
| `first_order_only` | | | | |
| `second_order_only` | | | | |
| `shuffle_higher_orders` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared slice lift >= +0.04
- [ ] A1 (`first_order_only`) loses >= 70% of slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p024.

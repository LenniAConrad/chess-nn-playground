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
- Primitive: Event-Delta Bilinear Accumulator (EDBA)
- `primitive_gate` mean / max / fraction > 0.5
- `edba_first_order_magnitude` distribution
- `edba_pair_term_magnitude` distribution
- Correlation: `edba_pair_term_magnitude` vs label

## Slice Findings

- Declared target slice: second-order interaction tactics (king-piece
  distance, bishop pair, mutual defense, overloaded defender)
  - Required: p022 unablated >= i193 + 0.04 PR AUC
  - Required: A1 (`first_order_only`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | mean Q magnitude | gate mean |
|---|---|---|---|---|
| `none` | | | | |
| `first_order_only` | | | | |
| `shuffle_pair_term` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared slice lift >= +0.04
- [ ] A1 (`first_order_only`) loses >= 70% of slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p022.

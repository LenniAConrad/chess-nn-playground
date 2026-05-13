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
- Primitive: Reversible Delta Kernel Memory (RDKM)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positive samples
  - Negative near-puzzle samples
- `primitive_delta` distribution on the same two buckets
- `rdkm_active_count` vs label correlation
- `rdkm_memory_norm` and `rdkm_z_norm` distribution

## Slice Findings

- Declared target slice: piece-piece interaction patterns (pinned-
  piece-plus-pinner / king-piece-distance / overloaded defender)
  - Required: p019 unablated >= i193 + 0.04 PR AUC on slice
  - Required: A1 (`shuffle_tokens`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean on positives | gate mean on negatives |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_tokens` | | | | |
| `zero_memory` | | | | |
| `uniform_query` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.04
- [ ] A1 (`shuffle_tokens`) loses >= 70% of the slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p019.

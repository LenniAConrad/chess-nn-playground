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
- Primitive: Gibbs Cut Log-Partition Operator (GCLP)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positive samples
  - Negative near-puzzle samples
- `primitive_delta` distribution on the same two buckets
- `gibbs_log_partition_mean` and `_max` distributions
- `gibbs_cut_edge_energy` per-sample mean correlation with king safety

## Slice Findings

- Declared target slice: king-safety / fortress positions
  - Required: p037 unablated >= i193 + 0.03 PR AUC on slice
  - Required: A1 (`shuffle_logpartition`) loses >= 70% of that lift
- Watch slice: late-game endgame fortress positions
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean on positives | gate mean on negatives |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_logpartition` | | | | |
| `uniform_edges` | | | | |
| `uniform_sources` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.03
- [ ] A1 (`shuffle_logpartition`) loses >= 70% of the slice lift
- [ ] A2 (`uniform_edges`) loses >= 40% of the slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p037.

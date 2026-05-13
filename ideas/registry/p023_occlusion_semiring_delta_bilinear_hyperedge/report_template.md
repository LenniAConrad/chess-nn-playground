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
- Primitive: Occlusion Semiring Delta-Bilinear Hyperedge (OSDB)
- `primitive_gate` mean / max / fraction > 0.5
- `osdb_hidden_magnitude` distribution
- `osdb_pair_hyperedge_magnitude` distribution

## Slice Findings

- Declared target slice: through-the-square motifs (pin, skewer,
  x-ray, battery along one line)
  - Required: p023 unablated >= i193 + 0.04 PR AUC
  - Required: A1 (`disable_bilinear`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | mean hyperedge magnitude | gate mean |
|---|---|---|---|---|
| `none` | | | | |
| `disable_bilinear` | | | | |
| `zero_occupancy` | | | | |
| `uniform_occupancy` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared slice lift >= +0.04
- [ ] A1 (`disable_bilinear`) loses >= 70% of slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p023.

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
- Primitive: Differentiable Occupancy Eikonal Transform (OET)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positive samples
  - Negative near-puzzle samples
- `primitive_delta` distribution on the same two buckets
- `eikonal_field_mean`, `eikonal_field_max`, `eikonal_field_min`,
  `eikonal_field_range` distributions

## Slice Findings

- Declared target slice: king-safety / escape-corridor positions
  - Required: p039 unablated >= i193 + 0.03 PR AUC on slice
  - Required: A1 (`shuffle_field`) loses >= 70% of that lift
  - Required: A2 (`single_iteration`) loses >= 30% of that lift
- Watch slice: positions with no king under attack
- Near-puzzle FP rate at matched recall
- Per-slice breakdowns required (must not regress vs i193):
  - `crtk_difficulty` buckets (easy / medium / hard) — lift should
    concentrate on the medium / hard buckets without regressing the
    easy bucket
  - `crtk_phase` buckets (opening / middlegame / endgame) — lift
    should hold on middlegame / endgame king-safety positions, with
    no opening-bucket regression
  - `crtk_eval_bucket`, `crtk_tactic_motifs`, `crtk_tag_families`
- Per-slice false-positive rate for fine label `1` and false-negative
  rate for fine label `2`, jointly stratified by `crtk_difficulty` x
  `crtk_phase`.
- Highest-confidence wrong examples must report FEN,
  `crtk_difficulty`, `crtk_phase`, and motifs.

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean | field range |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_field` | | | | |
| `single_iteration` | | | | |
| `uniform_costs` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.03
- [ ] A1 (`shuffle_field`) loses >= 70% of the slice lift
- [ ] A2 (`single_iteration`) loses >= 30% of the slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p039.

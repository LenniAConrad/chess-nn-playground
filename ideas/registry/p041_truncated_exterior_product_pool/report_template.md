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
- Primitive: Truncated Exterior Product Pool (TEPP)
- `primitive_gate` mean / max / fraction > 0.5 on positives / negatives
- `primitive_delta` distribution
- `tepp_grade_<k>_magnitude` per grade `k = 0..R`
- `tepp_grade_max_magnitude`, `tepp_grade_mean_magnitude`

## Slice Findings

- Declared target slice: `tepp_grade_2_magnitude` above median (positions
  where the wedge cancellation is informative)
  - Required: p041 unablated >= i193 + 0.03 PR AUC on slice
  - Required: A1 (`shuffle_grades_high`) loses >= 70% of that lift
  - Required: A2 (`first_order_only`) loses >= 40% of that lift
- Watch slice: positions with few active pieces (grade-3 magnitude
  must remain finite)
- Near-puzzle FP rate at matched recall
- Per-slice breakdowns required (must not regress vs i193):
  - `crtk_difficulty` buckets (easy / medium / hard) — wedge
    cancellation is expected to lift the medium / hard buckets where
    multi-piece interactions dominate, without regressing the easy
    bucket
  - `crtk_phase` buckets (opening / middlegame / endgame) — lift
    should concentrate on middlegame positions where higher-grade
    monomials still survive truncation, with no opening-bucket
    regression
  - `crtk_eval_bucket`, `crtk_tactic_motifs`, `crtk_tag_families`
- Per-slice false-positive rate for fine label `1` and false-negative
  rate for fine label `2`, jointly stratified by `crtk_difficulty` x
  `crtk_phase`.
- Highest-confidence wrong examples must report FEN,
  `crtk_difficulty`, `crtk_phase`, and motifs.

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean | grade-2 magnitude |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_grades_high` | | | | |
| `first_order_only` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.03
- [ ] A1 (`shuffle_grades_high`) loses >= 70% of the slice lift
- [ ] A2 (`first_order_only`) loses >= 40% of the slice lift
- [ ] Throughput drop versus i193 < 30%

If any box fails: drop p041.

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

## Coverage Diagnostics

- Mechanism family: `convex`
- Ablation: `none`
- Coverage score `F(a)` (mean / by label):
- Coverage vector `c` saturation by label:
- Top marginal gains (validation examples):
- Active concept count (mean / by label):
- Saturation gap (coverage − additive pool):
- Max marginal gain:
- Concept entropy:
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Submodular Coverage Bottleneck` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior, plus the coverage diagnostics above.

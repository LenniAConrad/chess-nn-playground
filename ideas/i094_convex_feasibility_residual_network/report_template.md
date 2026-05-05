# Report Template

## Run

- Result path:
- Config:
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Packet Diagnostics

- Mechanism family: `robustness`
- Packet auxiliary logit:
- Mechanism energy:
- Sheaf tension / transport imbalance / symmetry residual / topology pressure as applicable:
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Convex Feasibility Residual Network` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior.

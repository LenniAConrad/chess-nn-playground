# Report Template

## Run

- Result path:
- Config:
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/all_ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Independence Residual Diagnostics

- Residual L1/L2:
- Positive / negative residual mass:
- Maximum absolute residual:
- Expected-map entropy:
- Piece / square / rank / file entropy:
- Rank/file coupling:
- Channel interaction energy:
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Independence Residual Interaction Network` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior.

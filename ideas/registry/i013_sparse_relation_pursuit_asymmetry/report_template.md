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

## Sparse Diagnostics

- Mean background residual:
- Mean tactical residual:
- Residual asymmetry by true class:
- Active atom fraction:
- Active group fraction:
- Dictionary coherence:
- Branch separation:
- Dead group penalty:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether SRPA is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and near-puzzle false-positive behavior.

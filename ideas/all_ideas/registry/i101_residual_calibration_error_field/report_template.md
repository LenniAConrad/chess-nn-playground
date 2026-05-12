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

## Calibration Diagnostics

- Raw logit quality:
- Calibration temperature distribution:
- Bounded correction norm:
- Confidence shift:
- Error-field energy / peak / entropy:
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Residual Calibration Error Field` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior.

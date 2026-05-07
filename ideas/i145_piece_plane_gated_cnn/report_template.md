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

## Model Diagnostics

- White/black/state gate means:
- Gate entropy:
- White and black piece counts:
- State signal:
- Semantic grouping known:
- Trunk feature energy and pooled feature spread:
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Piece-Plane Gated CNN` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior.

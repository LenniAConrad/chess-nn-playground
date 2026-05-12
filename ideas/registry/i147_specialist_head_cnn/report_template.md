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

## Specialist Diagnostics

- Global / center / edge / king / material head logits:
- Active head count:
- Learned fusion logit versus uniform logit average:
- Specialist logit shares:
- King-zone decode rate:
- Material balance and phase calibration:
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Specialist-Head CNN` is kept, refined, scaled, or rejected. The
decision must cite both aggregate metrics and specialist-head diagnostics.

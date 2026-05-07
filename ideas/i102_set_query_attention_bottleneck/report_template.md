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

## Attention Bottleneck Diagnostics

- Mechanism family: `graph`
- Attention entropy:
- Best-second attention margin:
- Occupied versus empty attention mass:
- Side-to-move versus opponent piece mass:
- Query diversity:
- Ablation comparison:
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Set-Query Attention Bottleneck` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior.

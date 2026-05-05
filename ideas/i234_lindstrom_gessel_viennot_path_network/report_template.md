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

- Mechanism family: `linear_algebra`
- Packet profile: `lindstrom_gessel_viennot_path_network`
- Packet auxiliary logit:
- Mechanism energy:
- Linear-algebra diagnostics (rank/spectral/moment summaries):
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Lindstrom-Gessel-Viennot Path Determinant Network` is kept, refined, scaled, or rejected. The decision
must cite both aggregate metrics and slice behavior.

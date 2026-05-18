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

## Packet Diagnostics

- Mechanism family: `spatial_cnn`
- Packet auxiliary logit:
- Mechanism energy (`mechanism_energy`):
- FPN level energies (`fpn_y8_energy`, `fpn_y4_energy`, `fpn_x2_energy`):
- Top-down update energies (`topdown_4_energy`, `topdown_8_energy`):
- Board FPN ablation code (`board_fpn_ablation`):
- Near-puzzle false positives:

## Ablation Deltas

- `none` vs `single_resolution_matched`:
- `none` vs `bottom_up_only`:
- `none` vs `no_2x2_level`:
- `none` vs `late_pool_only`:
- `none` vs `no_coordinate_planes`:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Board FPN CNN` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior, including whether the `single_resolution_matched` central falsifier was beaten.

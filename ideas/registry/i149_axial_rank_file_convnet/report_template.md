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
- Axial branch energies (`rank_energy`, `file_energy`, `local_energy`):
- Axial balance (`axial_balance`):
- Rank-file imbalance (`rank_file_imbalance`):
- Axial ablation code (`axial_rank_file_ablation`):
- Near-puzzle false positives:

## Ablation Deltas

- `none` vs `local_only`:
- `none` vs `rank_only`:
- `none` vs `file_only`:
- `none` vs `no_residual`:
- `none` vs `single_block`:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Axial Rank-File ConvNet` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior, including whether the `local_only` central falsifier was beaten.

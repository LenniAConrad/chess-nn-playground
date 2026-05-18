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

- Mechanism family: `linear_algebra`
- Packet auxiliary logit:
- Mechanism energy (`mechanism_energy`):
- Bispectral phase norm (`bispectral_phase_norm`):
- Bispectral magnitude mean (`bispectral_magnitude_mean`):
- Power spectrum energy (`power_spectrum_energy`):
- Cross-channel phase norm (`cross_phase_norm`):
- Bispectral ablation code (`bispectral_ablation`):
- Near-puzzle false positives:

## Ablation Deltas

- `none` vs `magnitude_only`:
- `none` vs `power_only`:
- `none` vs `phase_batch_shuffle`:
- `none` vs `random_frequency_pairs`:
- `none` vs `channel_pair_shuffle`:
- `none` vs `no_coordinate_planes`:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Bispectral Phase-Coupling Board Network` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior, including whether the `magnitude_only` and `power_only` central falsifiers were beaten.

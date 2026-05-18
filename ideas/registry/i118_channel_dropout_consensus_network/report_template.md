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

## Consensus Diagnostics

- Mechanism family: `robustness`
- Ablation code (`channel_dropout_ablation`):
- View count (`channel_dropout_view_count`):
- Consensus energy (`consensus_energy`):
- Disagreement energy (`disagreement_energy`):
- Max pairwise energy (`max_pairwise_energy`):
- Full-view energy (`full_view_energy`):
- Mechanism energy (`mechanism_energy`):
- Near-puzzle false positives:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Ablation Comparison

For each of `none`, `full_view_only`, `mean_only`, `random_channel_masks`, `train_dropout_only` (one row per ablation):

- AUROC / balanced accuracy / F1 / calibration vs. `none`.
- Delta on near-puzzle false-positive rate vs. `none`.
- Delta on `consensus_energy` and `disagreement_energy` vs. `none`.

## Decision

State whether `Channel Dropout Consensus Network` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior, and must reference the central ablation deltas (`full_view_only` and `train_dropout_only` are the load-bearing comparisons; `random_channel_masks` is the falsification check).

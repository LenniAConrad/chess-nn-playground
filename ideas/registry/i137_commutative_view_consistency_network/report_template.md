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

## Consistency Diagnostics

- Mechanism family: `symmetry`
- Ablation code (`commutative_view_ablation`):
- View count (`commutative_view_count`):
- Consistency energy (`consistency_energy`):
- Mean defect L1 (`mean_defect_l1`):
- Mean defect cosine (`mean_defect_cosine`):
- Per-defect L2 norms (`defect_l2`):
- Per-defect L1 norms (`defect_l1`):
- Per-view RMS norms (`view_norms`):
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

For each of `none`, `views_only_no_defects`, `single_square_view`, `random_view_maps`, `count_to_all_only`, `shuffled_piece_view` (one row per ablation):

- AUROC / balanced accuracy / F1 / calibration vs. `none`.
- Delta on near-puzzle false-positive rate vs. `none`.
- Delta on `consistency_energy`, `mean_defect_l1`, `mean_defect_cosine` vs. `none`.

## Decision

State whether `Commutative View-Consistency Network` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior, and must reference the central ablation deltas (`views_only_no_defects` and `single_square_view` are the load-bearing comparisons; `random_view_maps` is the falsification check for the learned cross-view maps; `count_to_all_only` and `shuffled_piece_view` are material-shortcut and piece-geometry checks).

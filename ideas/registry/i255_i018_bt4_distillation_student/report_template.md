# Report Template

## Run

- Result path:
- Config:
- Cell (scale x encoding x loss tranche x falsifier):
- Seeds (typically 42 / 43 / 44):
- GPU:
- Training budget: `epochs=20`, `min_epochs=10`,
  `early_stopping_patience=5`, `monitor=pr_auc`, `batch_size=192`,
  `reduce_on_plateau (factor=0.5, patience=2)`
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`
- Paired i018 teacher baseline path (same split, same seeds,
  `scale_xl`):
- Paired BT4 supervised baseline path (`lc0_bt4` at same channels):
- Paired ablation row (one tranche off):

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC (paper-grade primary):
- Calibration:

## Distillation Diagnostics

- Loss config (`bce_with_logits` baseline, or `i018_bt4_distill` with
  per-term weights `lambda_sup / kd / diag / plane / read / brier / rank`):
- Teacher temperature `T_t` fit on validation:
- Student post-hoc temperature on validation:
- Brier score before / after student temperature scaling:
- ECE before / after student temperature scaling:
- Per-head training loss curves (`logits`, `diagnostic`, `plane`,
  `readout`) saved to:

## Architecture Diagnostics

- Mechanism family: `distillation` (BT4 conv backbone + i018 targets)
- Backbone shape (`channels`, `num_blocks`, `input_channels`,
  `encoding`):
- Auxiliary heads active (`diagnostic_dim`, `summary_plane_dim`,
  `readout_dim`):
- Canonicalization on/off:
- Total parameters:

## Promotion Gate

| Metric                              | base target | scale_up target | this run |
|-------------------------------------|------------:|----------------:|---------:|
| PR-AUC                              | >= 0.875    | >= 0.880        |          |
| Near-puzzle FP @ recall 0.80        | <= 0.16     | <= 0.155        |          |
| Puzzle recall                       | >= 0.80     | >= 0.80         |          |
| Batch-1 CPU latency                 | <= 1.2 ms   | <= 1.6 ms       |          |

The student is promoted only if every row of the gate clears for the
tier (`base` or `scale_up`) it is intended to ship at.

## Falsifier Outcomes

- F1 (summary planes shuffled across batch): PR-AUC drop vs intact A5:
- F2 (diagnostics shuffled across batch):    PR-AUC drop vs intact A3:
- F3 (readout replaced by random vectors):   PR-AUC drop vs intact A6:
- F4 (`lambda_kd = 0` inside the full stack): PR-AUC delta vs
  supervised BT4 baseline (should be ~0):

A falsifier that does NOT drop PR-AUC means the corresponding loss
term was decorative; drop it from the production loss before promotion.

## Slice Findings

Summarise performance by, with paired BT4 supervised and paired i018
teacher baselines as columns:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

Per-slice false positives for fine label `1`, per-slice false
negatives for fine label `2`, confidence/calibration by slice, and the
highest-confidence wrong examples (FEN, difficulty, phase, motifs) are
all required by `ideas/docs/BENCHMARK_REPORTING.md`.

## Keep / Drop Decision

- [ ] At least one ablation row lifts PR-AUC by `+0.005` vs the
      supervised BT4 baseline (A1).
- [ ] Same row also cuts near-puzzle false positives at recall 0.80
      by `1%` vs supervised BT4 baseline.
- [ ] No slice regression on hard / equal / endgame / mate_in_1 /
      promotion / underpromotion vs supervised BT4 baseline.
- [ ] Batch-1 CPU latency under the tier gate.
- [ ] F4 (`lambda_kd = 0`) collapses the headline distillation effect
      (sanity check that the loss is doing the work).

If any box fails, do not promote that distilled row.

## Conclusions

- Does richer-than-logits distillation from i018 close the gap to
  i018 `scale_xl` at BT4-class CPU latency? (yes / no)
- Which loss term (if any) is doing the work? (KD / diag / plane /
  read / rank)
- Was scaling the student up worth it, or did `base` suffice?
- Highest-confidence wrong examples on the winning row (FEN,
  difficulty, phase, motifs):
- Recommended next step (ship base, ship scale_up, run the
  near-puzzle-emphasis tranche, run the multi-teacher ablation with
  slice specialists such as i024 or i193, or drop distillation in
  favour of teacher-of-record):

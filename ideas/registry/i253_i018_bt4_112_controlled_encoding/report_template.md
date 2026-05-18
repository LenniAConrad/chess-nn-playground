# Report Template

## Run

- Result path:
- Config:
- Cell (encoding x relation_mode x falsifier):
- Seeds (typically 42 / 43 / 44):
- GPU:
- Training budget: `epochs=30`, `min_epochs=15`,
  `early_stopping_patience=8`, `monitor=pr_auc`, `batch_size=192`,
  `reduce_on_plateau (factor=0.5, patience=2)`
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`
- Paired i018 baseline path (same split, same seeds, base scale):
- Paired simple18-control row (matched relation_mode):
- Paired falsifier row (scramble and/or augmentation_only):

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC (paper-grade primary):
- Calibration:

## Controlled Encoding Diagnostics

- Encoding (`simple_18` / `lc0_bt4_112`):
- relation_mode (`exact` / `confidence` / `hybrid`):
- augmentation_lambda:
- scramble_exact_relations:
- augmentation_only:
- input_channels at the controlled raw branch (must be 112 for both
  encodings):
- Total parameters (must match the matched-budget table:
  exact 94,371 / confidence 99,487 / hybrid 102,763 at the base scale):

## Packet Diagnostics

- Mechanism family: `sheaf` (inherited from i018)
- Sheaf tension / transport imbalance / per-relation energy /
  triad-defect / pin pressure / king-ring summaries:
- `controlled_confidence_mean` (confidence/hybrid only):
- `controlled_augmentation_mean` (hybrid only):
- Near-puzzle false positives at matched recall:

## Falsifier Outcomes

- Relation scramble PR-AUC drop vs intact same-row baseline:
- Hybrid scramble PR-AUC drop vs intact hybrid same-row baseline:
- Augmentation-only PR-AUC gap vs intact hybrid same-row baseline:

These must clear the i018 thresholds: scramble drop at least `0.02`
PR-AUC; augmentation-only must collapse below intact hybrid.

## Slice Findings

Summarise performance by, with paired simple18 and paired i018 baselines
as columns:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

Per-slice false positives for fine label `1`, per-slice false negatives
for fine label `2`, confidence/calibration by slice, and the highest-
confidence wrong examples (FEN, difficulty, phase, motifs) are all
required by `ideas/docs/BENCHMARK_REPORTING.md`.

## Keep / Drop Decision

- [ ] Relation scramble dropped at least `0.02` PR-AUC vs the matched
      intact row (BT4 and simple18, every relation_mode tested).
- [ ] Hybrid augmentation-only is clearly below intact hybrid (the
      augmentation head is a residual, not a replacement).
- [ ] At least one BT4 row beats its matched simple18 row by `+0.003`
      PR-AUC, or near-puzzle false positives at matched recall drop by
      `1%`, with no slice regression on hard / equal / endgame /
      mate-in-1 / promotion / underpromotion.
- [ ] Effect persists across seeds 42 / 43 / 44.

If any box fails the corresponding row, do not promote that BT4 row;
keep simple18 as the canonical i018 input.

## Conclusions

- Does the current single-FEN BT4 exporter add usable signal to the
  i018 trunk under preserved exact relation geometry? (yes / no)
- Which `relation_mode` (if any) carries the BT4 effect?
  (`exact` / `confidence` / `hybrid` / none)
- Was the augmentation head load-bearing or did exact masks dominate?
- Highest-confidence wrong examples on the winning row (FEN,
  difficulty, phase, motifs):
- Recommended next step (promote a BT4 row, scale i253 to xl, run with
  the i249 execution path for wall-clock, or drop the BT4 input choice
  for i018):

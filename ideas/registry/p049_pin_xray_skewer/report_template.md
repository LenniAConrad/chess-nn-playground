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

## Architecture-Specific Diagnostics

- Mechanism family: `king_path`
- Primitive: PXS (Pin / X-ray / Skewer)
- ``pxs_event_total_mean`` distribution
- ``pxs_abs_pin_mean`` / ``pxs_abs_pin_max`` distributions
- ``pxs_rel_pin_mean`` / ``pxs_rel_pin_max`` distributions
- ``pxs_xray1_mean`` / ``pxs_xray1_max`` distributions
- ``pxs_discovered_mean`` / ``pxs_discovered_max`` distributions
- ``pxs_skewer_mean`` / ``pxs_skewer_max`` distributions
- ``pxs_pinned_defender_mean`` / ``pxs_pinned_defender_max`` distributions
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-event positions (`pxs_event_total_mean` > median)
  - sparse-piece / endgame positions (`pxs_event_total_mean` < median)
- `primitive_delta` distribution on the same two buckets

## Slice Findings

- Target slices: `pin`, `skewer`, `discovered_attack`
  - Required: p049 unablated >= i193 + 0.01 PR AUC on at least one
    of those three slices
  - Required: A1 (`no_xray1`) loses >= 50% of that lift
  - Required: A2 (`uniform_values`) loses >= 30% of that lift
  - Required: A4 (`shuffle_rays`) loses essentially all of the lift
- Per-slice breakdowns required (must not regress vs i193):
  - `crtk_difficulty` buckets (easy / medium / hard) -- sliding
    motifs are expected to lift the medium / hard buckets where
    long-range tactics dominate, without regressing the easy bucket.
  - `crtk_phase` buckets (opening / middlegame / endgame) -- lift
    should concentrate on middlegame positions where slider tactics
    dominate, with no opening-bucket regression.
  - `crtk_eval_bucket`, `crtk_tactic_motifs`, `crtk_tag_families`.
- Per-slice false-positive rate for fine label `1` and false-negative
  rate for fine label `2`, jointly stratified by `crtk_difficulty` x
  `crtk_phase`.
- Highest-confidence wrong examples must report FEN,
  `crtk_difficulty`, `crtk_phase`, and motifs.

## Ablation Comparison Table

| Ablation | pin PR AUC | skewer PR AUC | disc PR AUC | aggregate PR AUC | gate mean |
|---|---|---|---|---|---|
| `none` | | | | | |
| `no_xray1` | | | | | |
| `uniform_values` | | | | | |
| `no_pin_def` | | | | | |
| `shuffle_rays` | | | | | |
| `zero_delta` | | | | | |
| `trunk_only` | | | | | |
| `disable_gate` | | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] At least one of {pin, skewer, discovered_attack} lifts >= +0.01
- [ ] A1 (`no_xray1`) loses >= 50% of the lift
- [ ] A2 (`uniform_values`) loses >= 30% of the lift
- [ ] A4 (`shuffle_rays`) loses essentially all of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p049 (or revisit the defender-load proxy and
the value-field initialisation).

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote to native i018 integration / drop):

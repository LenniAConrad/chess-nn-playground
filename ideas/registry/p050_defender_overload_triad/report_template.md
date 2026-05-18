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

- Mechanism family: `defender_load`
- Primitive: DOT (Defender Overload Triad)
- ``overload_operator_mean`` / ``overload_operator_l2`` distributions
- ``overload_us_mean`` / ``overload_us_peak`` distributions
- ``overload_them_mean`` / ``overload_them_peak`` distributions
- ``overload_defender_burden_us`` / ``overload_defender_burden_them`` distributions
- ``overload_pinned_share_us`` / ``overload_pinned_share_them`` distributions
- ``overload_criticality_us`` / ``overload_criticality_them`` distributions
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-overload positions (`overload_operator_l2` > median)
  - sparse-piece / endgame positions (`overload_operator_l2` < median)
- `primitive_delta` distribution on the same two buckets

## Slice Findings

- Target slices: `overload`, `pin`, `deflection`
  - Required: p050 unablated >= i193 + 0.01 PR AUC on at least one
    of those three slices
  - Required: A1 (`no_cross_target_load`) loses >= 50% of that lift
  - Required: A2 (`no_pins`) loses >= 20% of the lift on `pin`
  - Required: A3 (`no_target_value`) loses >= 30% of the lift on
    at least `overload`
- Per-slice breakdowns required (must not regress vs i193):
  - `crtk_difficulty` buckets (easy / medium / hard) -- defender-
    reuse motifs are expected to lift the medium / hard buckets
    where tactical density is higher, without regressing the easy
    bucket.
  - `crtk_phase` buckets (opening / middlegame / endgame) -- lift
    should concentrate on middlegame positions where overload and
    pin tactics dominate, with no opening-bucket regression.
  - `crtk_eval_bucket`, `crtk_tactic_motifs`, `crtk_tag_families`.
- Per-slice false-positive rate for fine label `1` and false-
  negative rate for fine label `2`, jointly stratified by
  `crtk_difficulty` x `crtk_phase`.
- Highest-confidence wrong examples must report FEN,
  `crtk_difficulty`, `crtk_phase`, and motifs.

## Ablation Comparison Table

| Ablation | overload PR AUC | pin PR AUC | deflection PR AUC | aggregate PR AUC | gate mean |
|---|---|---|---|---|---|
| `none` | | | | | |
| `no_cross_target_load` | | | | | |
| `no_pins` | | | | | |
| `no_target_value` | | | | | |
| `counts_only` | | | | | |
| `zero_delta` | | | | | |
| `trunk_only` | | | | | |
| `disable_gate` | | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] At least one of {overload, pin, deflection} lifts >= +0.01
- [ ] A1 (`no_cross_target_load`) loses >= 50% of the lift
- [ ] A2 (`no_pins`) loses >= 20% of the pin-slice lift
- [ ] A3 (`no_target_value`) loses >= 30% of the overload-slice lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p050 (or revisit the pin discount / king-ring
extension).

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote to native i018 integration / drop):

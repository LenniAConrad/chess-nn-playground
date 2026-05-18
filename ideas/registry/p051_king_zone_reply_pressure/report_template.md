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

- Mechanism family: `king_safety`
- Primitive: KZRP (King-Zone Reply Pressure)
- ``kzrp_operator_mean`` / ``kzrp_operator_l2`` distributions
- ``kzrp_asym_score`` distribution (us zone pressure minus them
  zone pressure) -- expected to skew positive on positive-class
  samples where the mover has a forcing king attack
- ``kzrp_us_zone_pressure`` / ``kzrp_them_zone_pressure``
  distributions
- ``kzrp_us_fake_defense_loss`` / ``kzrp_them_fake_defense_loss``
  distributions
- ``kzrp_us_live_escapes`` / ``kzrp_us_sealed_escapes`` /
  ``kzrp_us_blocked_escapes`` distributions
- ``kzrp_us_king_attack_mass`` distribution (current check severity
  proxy)
- ``kzrp_us_front_attack_mass`` distribution
- ``kzrp_us_reply_proxy`` distribution
- ``kzrp_us_ring_free_defense`` distribution
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-pressure positions (`kzrp_us_zone_pressure` > median)
  - sparse-piece / endgame positions (`kzrp_us_zone_pressure`
    near zero)
- `primitive_delta` distribution on the same two buckets

## Slice Findings

- Target slices: `mate_in_1`, near-puzzle FP at recall 0.8,
  `discovered_attack`
  - Required: p051 unablated >= i193 + 0.005 PR AUC on at least one
    of those three slices
  - Required: A1 (`no_front_zone`) loses >= 30% of that lift on
    at least `mate_in_1`
  - Required: A2 (`no_pins`) loses >= 20% of the lift on
    positions where the `pin` slice intersects king-zone activity
  - Required: A6 (`no_asymmetry`) loses >= 30% of the lift on
    `mate_in_1`
- Per-slice breakdowns required (must not regress vs i193):
  - `crtk_difficulty` buckets (easy / medium / hard) -- KZRP is
    expected to lift the hard bucket where attack density is
    higher, without regressing the easy bucket.
  - `crtk_phase` buckets (opening / middlegame / endgame) --
    lift should concentrate on middlegame positions where king
    attacks dominate, with no opening-bucket regression.
  - `crtk_eval_bucket`, `crtk_tactic_motifs`, `crtk_tag_families`.
- Per-slice false-positive rate for fine label `1` and false-
  negative rate for fine label `2`, jointly stratified by
  `crtk_difficulty` x `crtk_phase`.
- Highest-confidence wrong examples must report FEN,
  `crtk_difficulty`, `crtk_phase`, and motifs.

## Ablation Comparison Table

| Ablation | mate_in_1 PR AUC | near-puzzle FP @ recall 0.8 | discovered_attack PR AUC | aggregate PR AUC | gate mean |
|---|---|---|---|---|---|
| `none` | | | | | |
| `no_front_zone` | | | | | |
| `no_pins` | | | | | |
| `uniform_zone_weights` | | | | | |
| `no_escape_decomp` | | | | | |
| `uniform_units` | | | | | |
| `no_asymmetry` | | | | | |
| `zero_delta` | | | | | |
| `trunk_only` | | | | | |
| `disable_gate` | | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] At least one of {mate_in_1, near-puzzle FP, discovered_attack} lifts >= +0.005
- [ ] A1 (`no_front_zone`) loses >= 30% of the mate_in_1 lift
- [ ] A2 (`no_pins`) loses >= 20% of the pin-intersect lift
- [ ] A6 (`no_asymmetry`) loses >= 30% of the mate_in_1 lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p051 (or revisit the front-rank scope / zone
weights / reply-proxy formulation).

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote to native i018 integration / BT4
  plane-augmentation phase 3 / drop):

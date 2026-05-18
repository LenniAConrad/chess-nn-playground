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

- Mechanism family: `ray_scan_geometry`
- Primitive: EROS (Efficient Ray Occlusion Scan)
- `eros_occupancy_density` distribution
- `eros_mobility_mean` distribution
- `eros_xray_pressure_mean` distribution
- `eros_visible_density` distribution
- `eros_first_blocker_rate` / `eros_second_blocker_rate` distributions
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-second-blocker positions (`eros_second_blocker_rate > median`)
  - low-second-blocker positions (`eros_second_blocker_rate <= median`)
  - sparse-piece endgames (`eros_occupancy_density < median`)
- `primitive_delta` distribution on the same three buckets

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted idea must
report aggregate metrics plus the fine-label diagnostic matrix,
`slice_report_val.md`, `slice_report_test.md`, and performance broken
down by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. Include per-slice false
positives for fine label `1`, per-slice false negatives for fine label
`2`, confidence/calibration by slice, and the highest-confidence wrong
examples with FEN, difficulty, phase, and motifs.

## Slice Findings

- Target slice: "second-blocker-dependent" tactical puzzles -- x-ray
  attacks, discovered-attack frames, soft pins.
  - Required: p054 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`first_only`) loses >= 50% of that lift
  - Required: A2 (`no_blocker_id`) loses >= 30% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Required `crtk_difficulty` breakdown: lift must concentrate on
  medium/hard buckets without regressing the easy bucket.
- Required `crtk_phase` breakdown: lift must hold on middlegame and
  endgame buckets, with no opening-bucket regression.
- Required `crtk_tactic_motifs` breakdown: `pin`, `skewer`,
  `discovered_attack`, and `xray` motifs must show the lift
  concentration; if not, the second-blocker channels are misnamed.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | second_blocker_rate |
|---|---|---|---|---|
| `none` | | | | |
| `first_only` | | | | |
| `no_blocker_id` | | | | |
| `uniform_occupancy` | | | | |
| `empty_occupancy` | | | | |
| `shuffle_occupancy` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`first_only`) loses >= 50% of the lift
- [ ] A2 (`no_blocker_id`) loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p054.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):

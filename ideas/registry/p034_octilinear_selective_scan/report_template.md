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

- Mechanism family: `ray_scan`
- Primitive: OSS (Octilinear Selective Scan)
- Per-direction ``oss_energy_<dir>`` distributions (E, W, N, S, NE,
  NW, SE, SW)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-piece-density positions
  - low-piece-density / endgame positions
- `primitive_delta` distribution on the same two buckets

## Slice Findings

- Target slice: long-range ray coordination tactics
  (queen-rook batteries, long-diagonal pins, X-rays, batteries)
  - Required: p034 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`single_direction`) loses >= 50% of that lift
  - Required: A2 (`fixed_transition`) loses >= 30% of that lift
- Watch slice: `crtk_eval_bucket = equal` — must not regress
- Difficulty breakdown: report PR AUC and gate mean per
  `crtk_difficulty` bucket (easy / medium / hard). Long-range
  ray-scan lift should concentrate on medium/hard buckets where
  state propagation along a ray captures more than 1-hop adjacency;
  easy slices must not regress.
- Phase breakdown: report PR AUC, gate mean, and per-direction
  ``oss_energy_<dir>`` distributions per `crtk_phase` bucket
  (opening / middlegame / endgame). The diagonal (NE / NW / SE /
  SW) and file (N / S) directions should dominate in middlegame
  and endgame long-range coordination; opening must not regress.
- Per-slice false positives for fine label `1` and false negatives
  for fine label `2`, sliced by `crtk_difficulty` and `crtk_phase`.
- Highest-confidence wrong examples reported with FEN,
  `crtk_difficulty`, `crtk_phase`, and `crtk_tactic_motifs`.
- Near-puzzle FP rate at matched recall.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | mean energy |
|---|---|---|---|---|
| `none` | | | | |
| `single_direction` | | | | |
| `fixed_transition` | | | | |
| `shuffle_features` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`single_direction`) loses >= 50% of the lift
- [ ] A2 (`fixed_transition`) loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] No regression on the easy `crtk_difficulty` bucket
- [ ] No regression on the opening `crtk_phase` bucket
- [ ] Throughput drop versus i193 < 50% (otherwise plan kernel upgrade)

If any box fails: drop p034 (or upgrade to a parallel-scan kernel).

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / kernel upgrade):

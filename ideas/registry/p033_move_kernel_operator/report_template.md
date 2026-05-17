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

- Mechanism family: `legal_graph`
- Primitive: MKO (Move-Kernel Operator)
- Per-type ``mko_norm_<name>`` distributions (knight, rank, file, diag,
  antidiag, king)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-piece-density positions
  - low-piece-density / endgame positions
- `primitive_delta` distribution on the same two buckets
- Correlation: per-type norm vs final correctness

## Slice Findings

- Target slice: long-range tactical positions
  (queen-rook coordination, long-diagonal pins)
  - Required: p033 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`shared_kernel`) loses >= 50% of that lift
  - Required: A2 (`scalar_per_type`) loses >= 30% of that lift
- Watch slice: `crtk_eval_bucket = equal` — must not regress
- Difficulty breakdown: report PR AUC and gate mean per
  `crtk_difficulty` bucket (easy / medium / hard). Long-range
  tactical lift should concentrate on medium/hard buckets where
  move-type weight sharing is the discriminator; easy slices must
  not regress.
- Phase breakdown: report PR AUC, gate mean, and per-type
  ``mko_norm_<name>`` distributions per `crtk_phase` bucket
  (opening / middlegame / endgame). The diag / antidiag / file
  types should dominate in middlegame and endgame queen-rook
  coordination; opening must not regress.
- Per-slice false positives for fine label `1` and false negatives
  for fine label `2`, sliced by `crtk_difficulty` and `crtk_phase`.
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | mko_norm_diag |
|---|---|---|---|---|
| `none` | | | | |
| `shared_kernel` | | | | |
| `scalar_per_type` | | | | |
| `shuffle_features` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`shared_kernel`) loses >= 50% of the lift
- [ ] A2 (`scalar_per_type`) loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] No regression on the easy `crtk_difficulty` bucket
- [ ] No regression on the opening `crtk_phase` bucket
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p033.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):

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

- Mechanism family: `response_constraint`
- Primitive: Canonical-Orbit Straight-Through Operator (COSTO)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positive samples
  - Negative near-puzzle samples
- `primitive_delta` distribution on the same two buckets
- `cost_chosen_orbit_index` distribution (which of {e,F,R,FR} wins)
- `cost_orbit_gap` mean / median / quartiles
- `cost_orbit_ties` distribution (true ties have value >= 2)
- `cost_residual_norm` distribution

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

- Declared target slice: positions with `cost_orbit_gap` above median
  (positions that have a strong preferred orientation)
  - Required: p036 unablated >= i193 + 0.03 PR AUC on slice
  - Required: A1 (`shuffle_canonical`) loses >= 70% of that lift
- Watch slice: starting position symmetry / file-mirror duplicates
- Required `crtk_difficulty` breakdown: lift should concentrate on
  medium/hard buckets without regressing the easy bucket.
- Required `crtk_phase` breakdown: lift should hold on middlegame and
  endgame buckets, with no opening-bucket regression.
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean on positives | gate mean on negatives |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_canonical` | | | | |
| `identity_only` | | | | |
| `fixed_choice` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.03
- [ ] A1 (`shuffle_canonical`) loses >= 70% of the slice lift
- [ ] A2 (`identity_only`) loses >= 50% of the slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p036.

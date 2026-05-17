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
- Primitive: Blocker-Reset Ray Scan (BRRS)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positions with at least one slider
  - Positions without sliders
- `primitive_delta` distribution on the same two buckets
- `brrs_occupancy_density` vs label correlation
- `brrs_ray_magnitude` distribution

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

- Declared target slice: sliding-piece-dependent (pin / skewer /
  discovered attack / rook-on-open-file / queen line into king zone)
  - Required: p020 unablated >= i193 + 0.04 PR AUC on slice
  - Required: A1 (`zero_blocker`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Required `crtk_difficulty` breakdown: lift must concentrate on
  medium/hard buckets without regressing the easy bucket.
- Required `crtk_phase` breakdown: lift must hold on middlegame and
  endgame buckets (where slider geometry dominates), with no
  opening-bucket regression.
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean on sliders | gate mean on non-sliders |
|---|---|---|---|---|
| `none` | | | | |
| `zero_blocker` | | | | |
| `uniform_blocker` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.04
- [ ] A1 (`zero_blocker`) loses >= 70% of the slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p020.

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
- Primitive: Event-Symmetric Interaction Accumulator (ESIA)
- `primitive_gate` mean / max / fraction > 0.5
- `esia_order_<r>_magnitude` distribution for r = 1..R
- Correlation: `esia_order_2_magnitude` vs label

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

- Declared target slice: higher-order interaction tactics (fork,
  double-attack, discovered-attack triple, knight fork)
  - Required: p024 unablated >= i193 + 0.04 PR AUC
  - Required: A1 (`first_order_only`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Required `crtk_difficulty` breakdown: lift must concentrate on
  medium/hard buckets without regressing the easy bucket.
- Required `crtk_phase` breakdown: lift must hold on middlegame and
  endgame buckets (where higher-order interactions dominate), with no
  opening-bucket regression.
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | mean E^{(2)} magnitude | gate mean |
|---|---|---|---|---|
| `none` | | | | |
| `first_order_only` | | | | |
| `second_order_only` | | | | |
| `shuffle_higher_orders` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared slice lift >= +0.04
- [ ] A1 (`first_order_only`) loses >= 70% of slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p024.

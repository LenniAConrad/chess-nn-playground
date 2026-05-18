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

- Mechanism family: `counterfactual`
- Primitive: PUGP (Promotion and Underpromotion Geometry)
- `pugp_total_count` distribution (mean, max, fraction > 0)
- `pugp_push_count` / `pugp_capL_count` / `pugp_capR_count` distributions
- `pugp_n_own_r1` / `pugp_n_opp_r1` distributions
- `pugp_queen_check_count` / `pugp_queen_zone_max` distributions
- `pugp_knight_fork_max` distribution
- `primitive_gate` mean / max / fraction > 0.5 on:
  - positions with `pugp_total_count > 0`
  - positions with `pugp_total_count == 0`
  - merged `crtk_tactic_motifs in {promotion, underpromotion}` slice
- `primitive_delta` distribution on the same buckets

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

- Target slice: merged `crtk_tactic_motifs in {promotion,
  underpromotion}`
  - Required: p052 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`pseudo_only`) loses >= 50% of that lift
  - Required: at least one of A2 (`no_capture`), A3 (`queen_only`),
    A4 (`no_attack_defense`) loses >= 30% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Internal diagnostic breakdown (since the public slice currently
  merges promotion and underpromotion): report metrics conditional on
  the `pugp_*` diagnostic buckets so the underpromotion-specific
  contribution is visible.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean (target) | pugp_total_count mean |
|---|---|---|---|---|
| `none` | | | | |
| `pseudo_only` | | | | |
| `no_capture` | | | | |
| `queen_only` | | | | |
| `no_attack_defense` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`pseudo_only`) loses >= 50% of the lift
- [ ] At least one of A2 / A3 / A4 loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p052.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):

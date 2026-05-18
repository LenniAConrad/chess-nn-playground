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

- Mechanism family: `legal_routing`
- Primitive: LMGDP (Legal-Move-Graph Pressure-Delta)
- `lmgdp_total_edge_count` distribution (mean, max, fraction > 0)
- `lmgdp_edge_count_{P, N, B, R, Q, K}` distributions
- `lmgdp_post_attack_value_mean_{P, N, B, R, Q, K}` distributions
- `lmgdp_capture_value_mean_{P, N, B, R, Q, K}` distributions
- `primitive_gate` mean / max / fraction > 0.5 on:
  - positions with `lmgdp_total_edge_count > 0`
  - positions with `lmgdp_total_edge_count == 0`
  - merged `crtk_tactic_motifs in {capture, check, mate,
    promotion}` slice
- `primitive_delta` distribution on the same buckets

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted idea must
report aggregate metrics plus the fine-label diagnostic matrix,
`slice_report_val.md`, `slice_report_test.md`, and performance
broken down by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. Include per-slice
false positives for fine label `1`, per-slice false negatives for
fine label `2`, confidence / calibration by slice, and the
highest-confidence wrong examples with FEN, difficulty, phase, and
motifs.

## Slice Findings

- Target slice: merged `crtk_tactic_motifs in {capture, check,
  mate, promotion}`
  - Required: p053 unablated >= i193 + 0.01 PR AUC on target slice
  - Required: A1 (`no_pressure_delta`) loses >= 50% of that lift
  - Required: at least one of A2 (`no_capture_value`), A3
    (`random_typed_edges`), A4 (`shared_target_pool`) loses >= 30%
    of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Diagnostic breakdown: report metrics conditional on
  `lmgdp_post_attack_value_mean_Q` and
  `lmgdp_capture_value_mean_*` deciles to show whether the head
  picks up on candidate moves with the largest projected pressure
  delta.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean (target) | lmgdp_total_edge_count mean |
|---|---|---|---|---|
| `none` | | | | |
| `no_pressure_delta` | | | | |
| `no_capture_value` | | | | |
| `random_typed_edges` | | | | |
| `shared_target_pool` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Target slice lift >= +0.01
- [ ] A1 (`no_pressure_delta`) loses >= 50% of the lift
- [ ] At least one of A2 / A3 / A4 loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p053.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase,
  motifs):
- Recommended next step (promote / drop):

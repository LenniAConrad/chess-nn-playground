# Report Template — p006 Move-Graph Router

## Run

- Result path:
- Config: `ideas/registry/p006_move_graph_router/config.yaml`
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
- Primitive: MGR (move-graph routing on rule-derived sparse adjacency)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - tactical positives (`crtk_tactic_motifs != none`)
  - quiet positions
- `primitive_delta` distribution on the same buckets
- Correlation: `mgr_edge_count` vs `primitive_gate` (positions with
  more legal moves should be more interesting to the head, if MGR is
  doing what it claims)
- Correlation: `primitive_delta` vs final correctness

## Slice Findings

- Target slice: `crtk_tactic_motifs in {pin, skewer, x_ray}` (legal-move
  graph-routed tactics)
  - Required: p006 >= i193 + 0.04 PR AUC
  - Required: `random_edges` ablation loses >= 70% of that lift
- Watch slice: aggregate FP rate at matched recall
- Watch slice: `crtk_eval_bucket = equal` — must not regress

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean on positives | gate mean on quiet |
|---|---|---|---|---|
| `none` | | | | |
| `random_edges` | | | | |
| `dense_edges` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.04
- [ ] `random_edges` ablation loses >= 70% of the lift
- [ ] `dense_edges` ablation does NOT outperform `none`
- [ ] Throughput drop versus i193 < 30%

If any box fails: drop p006 and defer.

## Conclusions

- What the operator appears able to learn:
- What it appears unable to learn:
- Highest-confidence wrong examples:
- Recommended next step:

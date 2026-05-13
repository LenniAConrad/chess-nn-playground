# Report Template — p009 Legal-Move-Graph Convolution

## Run

- Result path:
- Config: `ideas/registry/p009_legal_move_graph_delta/config.yaml`
- Seeds:
- GPU:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`

## Aggregate Metrics

- Accuracy / F1 / ROC AUC / PR AUC / Calibration:

## Architecture-Specific Diagnostics

- Mechanism family: `legal_routing`
- Primitive: LMGConv (typed legal-move-graph convolution)
- Per-type message norms (`lmgconv_msg_norm_{P,N,B,R,Q,K}`):
  distribution across the validation set; should reflect per-piece-
  type prevalence on the board.
- `lmgconv_edge_count` distribution vs `primitive_gate`.

## Slice Findings

- Target slice:
  `crtk_tactic_motifs in {knight_fork, rook_battery, bishop_pin}`.
- Watch slice: aggregate FP rate at matched recall.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | knight msg norm | rook msg norm |
|---|---|---|---|---|
| `none` | | | | |
| `random_typed_edges` | | | | |
| `shared_weight` | | | | |
| `no_normalization` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.03
- [ ] `random_typed_edges` ablation loses >= 70% of the lift
- [ ] `shared_weight` ablation loses >= 50% of the lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p009.

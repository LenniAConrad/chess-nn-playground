# Report Template — p008 Rule-Conditioned Sparse Attention (MobScan)

## Run

- Result path:
- Config: `ideas/registry/p008_rule_conditioned_sparse_attention/config.yaml`
- Seeds:
- GPU:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`

## Aggregate Metrics

- Accuracy / F1 / ROC AUC / PR AUC / Calibration:

## Architecture-Specific Diagnostics

- Mechanism family: `legal_routing`
- Primitive: MobScan (selective SSM over rule-derived legal-move DAG)
- `mobscan_gate_A_mean / B_mean / C_mean`: distribution per epoch and
  per ablation; flat distributions hint at collapsed selectivity.
- `mobscan_state_norm` distribution vs `primitive_gate`.

## Slice Findings

- Target slice: multi-step tactical motifs (deep mate threats, two-move
  forcing lines).
- Watch slice: aggregate FP rate at matched recall.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate_A mean | gate_C mean |
|---|---|---|---|---|
| `none` | | | | |
| `random_edges` | | | | |
| `dense_edges` | | | | |
| `untied_state` | | | | |
| `single_iteration` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.04
- [ ] `random_edges` ablation loses >= 70% of the lift
- [ ] `untied_state` ablation loses >= 50% of the lift
- [ ] `single_iteration` ablation loses >= 30% of the lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p008.

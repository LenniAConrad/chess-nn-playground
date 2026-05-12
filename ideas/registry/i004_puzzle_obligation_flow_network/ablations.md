# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `no_flow_solver` | Replace allocation residual with simple pooled obligation/resource tokens. |
| `shuffled_compatibility` | Preserve tensor shape but destroy legal compatibility. |
| `no_capacity_constraints` | Remove resource capacity limits. |
| `uniform_demand_capacity` | Remove learned demand and capacity. |
| `king_obligations_only` | Keep only king-related obligations. |
| `material_obligations_only` | Keep only target/recapture obligations. |
| `cnn_trunk_only` | Board trunk without obligation flow. |

## What Each Ablation Tests

- `no_flow_solver`: tests whether the allocation bottleneck matters.
- `shuffled_compatibility`: tests chess-specific resource-obligation matching.
- `no_capacity_constraints`: tests whether resource scarcity is the signal.
- `uniform_demand_capacity`: tests learned obligation strength.
- `king_obligations_only`: tests king-tactic dependence.
- `material_obligations_only`: tests material-tactic dependence.
- `cnn_trunk_only`: tests total value over a plain board encoder.

## Falsification Criteria

Reject or revise if:

```text
no_flow_solver matches full model
or shuffled_compatibility matches full model
or no_capacity_constraints matches full model
```

Also reject if the model improves recall only by increasing near-puzzle false positives; the central target is hard-negative discrimination.


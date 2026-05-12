# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `max_attacker_only` | Remove defender candidates and use max attacker score. |
| `mean_payoff_pool` | Use mean payoff instead of equilibrium. |
| `no_exploitability_features` | Remove exploitability diagnostics from readout. |
| `random_defenders` | Replace defender candidates with random legal-like tokens. |
| `shuffled_payoff_rows` | Destroy attacker/defender pairing semantics. |
| `solver_steps_1_3_5_8` | Test solver depth. |
| `board_trunk_only` | Remove equilibrium layer entirely. |

## What Each Ablation Tests

- `max_attacker_only`: tests whether defenders matter.
- `mean_payoff_pool`: tests equilibrium versus simple pooling.
- `no_exploitability_features`: tests whether game diagnostics matter.
- `random_defenders`: tests legal defender semantics.
- `shuffled_payoff_rows`: tests candidate pairing.
- `solver_steps_1_3_5_8`: tests whether equilibrium iteration matters.
- `board_trunk_only`: tests total value over static classification.

## Falsification Criteria

Reject if:

```text
max_attacker_only matches full model
or mean_payoff_pool matches equilibrium
or random_defenders match legal defenders
or board_trunk_only matches full model
```

Also reject if equilibrium value does not separate source classes or if the model only improves random-position rejection while near-puzzle false positives remain high.


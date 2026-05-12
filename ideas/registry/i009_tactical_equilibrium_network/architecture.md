# Architecture

## Architecture Description

The Tactical Equilibrium Network has four parts:

1. board trunk
2. attacker and defender candidate encoders
3. entropy-regularized matrix-game solver
4. puzzle readout from equilibrium diagnostics

The architecture is designed to be cheaper than proof-number search but more adversarial than a static board classifier.

## Input Format

```text
board: batch x 18 x 8 x 8
attackers: batch x max_attackers x attacker_features
defenders: batch x max_defenders x defender_features
relations: batch x max_attackers x max_defenders x relation_features
```

First candidate limits:

```text
max_attackers: 16
max_defenders: 24
```

## Forward Pass

```text
H = board_trunk(board)
A = attacker_encoder(attackers, H)
D = defender_encoder(defenders, H)
P = payoff_head(A, D, relations)
p, q, value, exploitability = game_solver(P)
stats = summarize(P, p, q, value, exploitability)
logit = puzzle_head([pool(H), stats])
```

## Solver

First version can use unrolled mirror descent:

```text
p = uniform over attackers
q = uniform over defenders
for step in solver_steps:
    p = softmax((P @ q) / tau_attack)
    q = softmax((-P.T @ p) / tau_defense)
value = p.T @ P @ q
```

This is differentiable and easy to ablate.

## Tensor Shapes

Suggested first version:

```text
token_dim: 96
relation_dim: 32
max_attackers: 16
max_defenders: 24
solver_steps: 5
payoff_matrix: batch x 16 x 24
output: batch x 1
```

## Output Heads

Primary:

```text
puzzle_logit
```

Diagnostics:

```text
equilibrium_value
attacker_entropy
defender_entropy
exploitability
top_attacker_indices
top_defender_indices
```

## Parameter Estimate

```text
1M to 3M parameters
```

## FLOP Estimate

The game layer is small:

```text
O(batch * solver_steps * max_attackers * max_defenders)
```

Most cost is in the board trunk and candidate encoders.

## Implementation Binding

- Registered model name: `tactical_equilibrium_network`.
- Source implementation: `src/chess_nn_playground/models/trunk/research_architectures.py`.
- Idea-local wrapper: `ideas/registry/i009_tactical_equilibrium_network/model.py`.

# Architecture

## Architecture Description

The Boundary-Edit Lagrangian Network has five parts:

1. board encoder
2. rule-bounded edit basis generator
3. unrolled edit-energy solver
4. dual energy readout for puzzle-making and puzzle-breaking
5. final puzzle classifier

The architecture should learn a boundary-distance representation, not only raw puzzle evidence.

## Input Format

```text
board: batch x 18 x 8 x 8
edit_basis:
  batch x max_edits x latent_dim
edit_features:
  batch x max_edits x edit_feature_dim
```

Edit basis families:

- tempo edit
- defender edit
- blocker edit
- line-opening edit
- target-protection edit
- king-escape edit
- relation-edge edit

## Forward Pass

```text
z = board_encoder(board)
base_logit = base_head(z)
edit_deltas, edit_costs = edit_basis_generator(board, z)
alpha_plus = unrolled_solver(z, edit_deltas, edit_costs, target="make_puzzle")
alpha_minus = unrolled_solver(z, edit_deltas, edit_costs, target="break_puzzle")
E_plus = energy(z, alpha_plus, target="make_puzzle")
E_minus = energy(z, alpha_minus, target="break_puzzle")
edit_stats = summarize(alpha_plus, alpha_minus, edit_costs)
logit = final_head([base_logit, E_minus - E_plus, E_plus, E_minus, edit_stats])
```

## Tensor Shapes

Suggested first version:

```text
latent_dim: 128
max_edits: 32
solver_steps: 4
edit_feature_dim: 32
output: batch x 1
```

## Output Heads

Primary:

```text
puzzle_logit
```

Diagnostics:

```text
base_logit
E_plus
E_minus
edit_gap
top_plus_edits
top_minus_edits
```

## Parameter Estimate

```text
1.5M to 4M parameters
```

## FLOP Estimate

The solver cost is:

```text
O(batch * solver_steps * max_edits * latent_dim)
```

This should be much cheaper than full multi-ply proof search.

## Implementation Binding

- Registered model name: `boundary_edit_lagrangian_network`.
- Source implementation: `src/chess_nn_playground/models/research_architectures.py`.
- Idea-local wrapper: `ideas/registry/i008_boundary_edit_lagrangian_network/model.py`.

# Architecture

## Architecture Description

The Puzzle Obligation Flow Network has four stages:

1. board trunk
2. obligation and resource candidate encoders
3. differentiable allocation solver
4. residual-flow puzzle head

The goal is to make the final logit depend on uncovered defensive obligations, not just raw attack pressure.

## Input Format

First implementation should use:

```text
board: batch x 18 x 8 x 8
```

All candidates are derived deterministically from the current board.

Candidate tensors:

```text
obligations: batch x max_obligations x obligation_features
resources: batch x max_resources x resource_features
compatibility_mask: batch x max_obligations x max_resources
```

No source labels or engine metadata are part of inference input.

## Forward Pass

```text
H = board_trunk(board)
O = encode_obligations(obligation_features, H)
R = encode_resources(resource_features, H)
demand = softplus(demand_head(O))
capacity = softplus(capacity_head(R))
compatibility = masked_bilinear(O, R, compatibility_mask)
P, duals = unrolled_flow_solver(demand, capacity, compatibility)
residual = relu(demand - P.sum(dim=resources))
z_flow = pool([residual, duals, O])
logit = puzzle_head([pool(H), z_flow])
```

## Candidate Types

Obligation types:

- king escape obligation
- check or latent-check answer obligation
- high-value target defense obligation
- slider-line block obligation
- pinned defender preservation obligation
- promotion/back-rank obligation

Resource types:

- king move
- capture attacker
- interpose piece
- recapture
- move defender
- counter-threat

## Tensor Shapes

Suggested first version:

```text
max_obligations: 32
max_resources: 48
token_dim: 96
solver_steps: 4
output: batch x 1
```

## Output Heads

Primary:

```text
puzzle_logit
```

Diagnostics:

```text
mean_residual_by_obligation_type
max_residual
dual_price_topk
allocation_entropy
```

Only `puzzle_logit` is used for benchmark prediction.

## Parameter Estimate

Rough first implementation:

```text
1M to 3M parameters
```

Most parameters are in the board trunk and token encoders.

## FLOP Estimate

Allocation bottleneck cost:

```text
O(batch * solver_steps * max_obligations * max_resources * token_dim)
```

This is heavier than a plain CNN but cheaper than full move/reply board re-encoding.

## Implementation Binding

- Registered model name: `puzzle_obligation_flow_network`.
- Source implementation: `src/chess_nn_playground/models/research_architectures.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i004_puzzle_obligation_flow_network/model.py`.

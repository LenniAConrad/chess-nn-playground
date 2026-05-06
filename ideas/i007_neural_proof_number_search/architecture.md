# Architecture

## Architecture Description

The model is a bounded differentiable proof-number search:

1. Encode the current board once.
2. Generate a deterministic tactical move beam.
3. Propagate latent states through a depth-limited AND/OR tree.
4. Predict proof and disproof costs at leaves.
5. Aggregate costs with differentiable proof-number rules.
6. Emit one puzzle logit from root proof/disproof statistics.

## Input Format

First implementation:

```text
board: batch x 18 x 8 x 8
move_tree:
  depth: 2 or 3
  beam_width_or: 8
  beam_width_and: 8
  move descriptors per edge
```

Move descriptors should include:

- from square
- to square
- moving piece type
- capture flag
- promotion flag
- gives-check-by-rule flag if available without engine search
- target value bucket
- relation to kings and high-value pieces

## Forward Pass

```text
z_root = board_encoder(board)
tree = generate_tactical_beam(board)
for depth in tree:
    z_child = transition(z_parent, move_descriptor, delta_features)
    node_score = tactical_node_head(z_child)
at leaves:
    proof_cost = softplus(proof_head(z_leaf))
    disproof_cost = softplus(disproof_head(z_leaf))
bottom-up:
    aggregate OR nodes with softmin proof and softsum disproof
    aggregate AND nodes with softsum proof and softmin disproof
logit = classifier([p_root, d_root, d_root - p_root, bounded_context])
```

## Tensor Shapes

Suggested first version:

```text
latent_dim: 128
or_beam: 8
and_beam: 8
depth: 3
max_nodes: about 1 + 8 + 64 + 512 before pruning
pruned_nodes: cap at 128-256 total
output: batch x 1
```

Keep the first implementation capped. The goal is to prove whether the mechanism helps, not to build a full chess search.

## Output Heads

Primary:

```text
puzzle_logit
```

Diagnostics:

```text
root_proof_cost
root_disproof_cost
proof_disproof_gap
best_proof_line_descriptors
tree_depth_used
beam_entropy
```

## Parameter Estimate

```text
2M to 6M parameters
```

depending on board encoder and transition depth.

## FLOP Estimate

The tree is the cost driver:

```text
O(batch * capped_nodes * latent_dim * transition_layers)
```

Do not re-run a CNN at every node in version 1.

## Implementation Binding

- Registered model name: `neural_proof_number_search`.
- Source implementation: `src/chess_nn_playground/models/research_architectures.py`.
- Idea-local wrapper: `ideas/i007_neural_proof_number_search/model.py`.

# Architecture

`Learnable Pooling Tree BoardNet` is a board-only classifier for the
`puzzle_binary` task. It accepts the repo's simple 18-plane current-board
tensor with shape `(B, 18, 8, 8)` and returns one puzzle logit per position.

The model builds a fixed hierarchy over the `8 x 8` board:

```text
squares (8x8) -> cells (4x4) -> quadrants (2x2) -> root (1x1)
```

Each pooling level groups four children in a `2 x 2` spatial neighbourhood and
collapses them into one parent. Pooling is **learnable**, not a fixed mean or
max: each tree node has a small aggregator that produces per-channel softmax
gate weights over its four children, mixes the children with those weights,
and applies an MLP transform plus a residual connection from the mean child.

Square features come from a coordinate-aware CNN stem on the board tensor:

```text
h_squares = SquareEncoder(concat(board, rank_plane, file_plane))
```

The pooling tree is then walked bottom-up:

```text
h_cells, gate_cell           = LearnablePool(h_squares)         # 8x8 -> 4x4
h_quadrants, gate_quadrant   = LearnablePool(h_cells)           # 4x4 -> 2x2
h_root, gate_root            = LearnablePool(h_quadrants)       # 2x2 -> 1x1
```

A FiLM-style top-down pass then broadcasts coarse context back down the tree,
so context flows from coarser tree levels back to finer ones:

```text
h_quadrants <- FiLM(h_root, h_quadrants)
h_cells     <- FiLM(h_quadrants, h_cells)
h_squares   <- FiLM(h_cells, h_squares)
```

The classifier reads pooled mean+max summaries from every tree level plus the
root vector:

```text
z      = concat(pool(h_squares), pool(h_cells), pool(h_quadrants),
                pool(h_root), root_vector)
logits = MLP(z)
```

Implemented ablations are `no_top_down`, `uniform_pool` (forces near-uniform
pooling weights by raising the softmax temperature, recovering a standard
average pool), and `single_level` (skips the top-down broadcasts).

Diagnostics include per-level pooled features, the learned pooling gate
weights at every level, gate entropy per level, per-level feature-energy
statistics, and a constant `tree_levels` marker.

## Implementation Binding

- Registered model name: `learnable_pooling_tree_boardnet`
- Source implementation file: `src/chess_nn_playground/models/learnable_pooling_tree_boardnet.py`
- Idea-local wrapper: `ideas/i164_learnable_pooling_tree_boardnet/model.py`

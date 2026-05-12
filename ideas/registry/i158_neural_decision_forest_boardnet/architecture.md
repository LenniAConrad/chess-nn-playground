# Architecture

`Neural Decision Forest BoardNet` realises the source packet's
"differentiable decision forest over CNN board features" thesis as a
bespoke architecture for the repo's `puzzle_binary` task. A compact
convolutional trunk produces a pooled board feature vector, and a soft
oblique forest with learned leaf logits returns the puzzle logit.

## Implementation Binding

- Registered model name: `neural_decision_forest_boardnet`
- Source implementation file: `src/chess_nn_playground/models/trunk/neural_decision_forest_boardnet.py`
- Idea-local wrapper: `ideas/registry/i158_neural_decision_forest_boardnet/model.py`

## Modules

`NeuralDecisionForestBoardNet` accepts the project's `(B, 18, 8, 8)`
board tensor only. CRTK / source / engine / verification metadata is
reporting-only and is not consumed.

1. **Board trunk.** `BoardForestTrunk` runs a ``3x3 Conv2d -> [BatchNorm2d ->] GELU``
   stem at width `trunk_width`, then `trunk_depth` ``BoardResidualBlock``
   blocks (two ``3x3`` convs with optional `BatchNorm2d`, GELU activations,
   and channel dropout). The board map is pooled by concatenating mean and
   max pooling, producing a ``2 * trunk_width`` summary, and projected by
   ``LayerNorm -> Linear -> GELU -> Dropout`` to the forest feature
   dimension `hidden_dim`. The pooled vector `z` is the input to every
   tree.
2. **Differentiable oblique forest.** `DifferentiableObliqueForest` holds
   `num_trees` balanced binary trees of depth `tree_depth`. Each tree has
   `2 ** tree_depth - 1` internal nodes and `2 ** tree_depth` leaves. A
   single shared `Linear(hidden_dim, num_trees * internal_nodes)` layer
   produces the oblique split logits for every node of every tree from
   `z`; right-branch probabilities are
   `sigmoid(split_logit / split_temperature)`. Per-leaf path
   probabilities are computed in closed form by indexing the sigmoid
   tensor with two precomputed buffers (`path_nodes`, `path_directions`)
   and multiplying along the tree depth, so a leaf is reached with
   probability ``prod_k [b_k * sigma + (1 - b_k) * (1 - sigma)]``.
3. **Leaf logits and forest average.** Leaf logits live in a
   ``(num_trees, leaf_count, num_classes)`` parameter `leaf_logits`.
   Each tree's prediction is the path-probability-weighted sum of its
   leaf logits, computed with
   ``einsum("btl,tlc->btc", path_probs, leaf_logits)``. The forest
   logit is the mean over the `num_trees` trees, squeezed to ``(B,)``
   for the `puzzle_binary` BCE-with-logits trainer when
   `num_classes == 1`.

## Forest Mathematics

For each input `z = trunk(x)`, tree `t`, internal node `n`, and leaf
`ell`:

```
g_{t,n}(z) = sigmoid( (w_{t,n} . z + b_{t,n}) / tau )
mu_{t, ell}(z) = prod_{k=0}^{d-1} [ b_k * g_{t, n_k}(z) + (1 - b_k) * (1 - g_{t, n_k}(z)) ]
y_t(z) = sum_{ell=1}^{L} mu_{t, ell}(z) * pi_{t, ell}
y(z)   = (1 / T) * sum_{t=1}^{T} y_t(z)
```

`tau = split_temperature` controls the sharpness of routing. The leaf
indexing (`path_nodes`, `path_directions`) is precomputed once at
construction; the forward pass is `O(B * T * (2^d - 1))` for routing
and `O(B * T * L * C)` for the leaf aggregation.

## Loss

The default trainer wires standard BCE-with-logits on
`output["logits"]`. There is no auxiliary loss; gradient flows through
the trunk, the per-node oblique-split layer, and every leaf logit on
every sample (no sparse routing).

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)` for `num_classes == 1` (BCE-compatible
  log-odds), `(B, num_classes)` otherwise.
- `prob`: shape `(B,)` sigmoid probability when `num_classes == 1`.
- `leaf_usage_entropy`: shape `(B,)`, mean across trees of the
  per-tree leaf-routing entropy normalised by `log(leaf_count)`.
- `per_tree_disagreement`: shape `(B,)`, variance of per-tree margins
  (zero when `num_trees == 1`).
- `dominant_leaf_index`: shape `(B,)`, mean (across trees) of the
  argmax leaf index per sample.
- `dominant_leaf_probability`: shape `(B,)`, mean (across trees) of
  the maximum leaf probability per sample.
- `path_probability_sum`: shape `(B,)`, mean across trees of the leaf
  probabilities sum (should be ~1).
- `mean_split_probability`: shape `(B,)`, mean over trees and
  internal nodes of the per-node sigmoid (a soft branch-balance score).
- `trunk_energy`: shape `(B,)`, mean square of the post-trunk feature
  map.
- `feature_energy`: shape `(B,)`, mean square of the pooled feature
  vector `z`.
- `split_feature_norm`: shape `(B,)`, broadcast scalar of the mean
  L2 norm of the split-layer rows (detached).
- `leaf_logit_norm`: shape `(B,)`, broadcast scalar of the leaf-logit
  Frobenius norm (detached).

## Contract

- Input: `(B, 18, 8, 8)` simple_18 board tensor only. Engine,
  verification, source, CRTK, principal-variation, mate-score, and
  best-move metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  `puzzle_binary` BCE-with-logits trainer, plus the diagnostics
  listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`;
  fine label `2` maps to binary target `1`.

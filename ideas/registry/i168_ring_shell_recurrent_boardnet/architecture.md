# Architecture

`Ring-Shell Recurrent BoardNet` is a board-only classifier for the
`puzzle_binary` task. It accepts the repo's simple 18-plane current-board
tensor with shape `(B, 18, 8, 8)` and returns one puzzle logit per position
together with the per-anchor / per-ring radial trail.

## Anchors

Important chess context radiates from a small set of anchors. The model
uses two *dynamic* anchors (the kings, located by soft-centroid of the
white-king and black-king planes) plus five *static* anchors that mark
center, promotion zones, and the queenside / kingside edges:

| Index | Name              | Position           | Source           |
|------:|-------------------|--------------------|------------------|
|     0 | `white_king`      | centroid of plane 5  | dynamic          |
|     1 | `black_king`      | centroid of plane 11 | dynamic          |
|     2 | `center`          | (3.5, 3.5)           | static           |
|     3 | `white_promotion` | (0.0, 3.5)           | static           |
|     4 | `black_promotion` | (7.0, 3.5)           | static           |
|     5 | `queenside_edge`  | (3.5, 0.0)           | static           |
|     6 | `kingside_edge`   | (3.5, 7.0)           | static           |

Static rings are pre-computed at construction; dynamic rings are rebuilt
per batch from the king centroids.

## Trunk

`Conv3x3 -> BatchNorm -> GELU` stem followed by `depth`
`ConvBlock`s (`Conv3x3 -> BatchNorm -> GELU -> Dropout`) of width
`channels`. The trunk emits a single feature map
`H \in R^{channels \times 8 \times 8}` shared across all anchors.

## Ring shell pooling

For anchor `i` and ring index `r = 0..R-1` (`R = num_rings`) the *Chebyshev
shell* mask is

```text
S_{i, r} = { (h, w) : floor( max(|h - row_i|, |w - col_i|) ) == r }
```

with `(row_i, col_i)` the anchor's position. The ring summary

```text
f_{i, r} = (1 / |S_{i, r}|) * sum_{(h, w) in S_{i, r}} H[:, h, w]
```

is the mean of the trunk feature map over the shell. A learned linear
projection plus a per-anchor bias produces the radial sequence

```text
\tilde f_{i, r} = W f_{i, r} + b_i
```

which is what the recurrent model consumes.

## Recurrent shell processor

A *single shared* `GRU` of hidden size `rnn_hidden` is run per anchor by
folding the anchor axis into the batch axis:

```text
h_{i, r} = GRU(\tilde f_{i, r}; h_{i, r-1})
```

The GRU integrates the radial sequence from the anchor outward, shell by
shell, and returns the final hidden state `h_i = h_{i, R-1}` per anchor.

## Head

Per-anchor final hidden states are concatenated and consumed by a
`LayerNorm -> Linear -> GELU -> Dropout -> Linear` MLP head whose output is
the puzzle logit:

```text
z = concat_i h_i
\hat{y} = Linear( GELU( Linear( LayerNorm(z) ) ) )
```

## Diagnostics

The forward pass returns a dict with `B = batch`, `A = num_anchors = 7`,
`R = num_rings`, `C = channels`, `H = rnn_hidden`:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` for `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `trunk_features`: `(B, C, 8, 8)` shared trunk feature map.
- `ring_pool`: `(B, A, R, C)` mean-pooled raw ring features.
- `ring_features`: `(B, A, R, C)` projected radial sequence fed to the GRU.
- `ring_energy`: `(B, A, R)` mean square of `ring_features` per ring.
- `ring_counts`: `(B, A, R)` number of squares belonging to each ring.
- `anchor_hidden`: `(B, A, H)` GRU final hidden state per anchor.
- `anchor_hidden_progression`: `(B, A, R, H)` GRU hidden state at every
  ring step, so downstream tooling can inspect how information flows from
  the anchor outward.
- `anchor_positions`: `(B, A, 2)` row/col of every anchor (the dynamic
  king positions update per batch).
- `anchor_dynamic_mass`: `(B, 2)` total mass of the white/black king
  plane; used to sanity-check that the dynamic anchors actually saw a
  king on the board.
- `anchor_dynamic_present`: `(B, 2)` boolean cast indicating which
  dynamic anchors had a non-zero king plane.
- `anchor_hidden_energy`: `(B, A)` mean square of each anchor's final
  hidden state.
- `mean_anchor_hidden_energy`: `(B,)` average over anchors of the above.
- `mean_ring_energy`: `(B,)` average ring-feature energy across all
  anchors and rings.
- `radial_progression_energy`: `(B, A, R)` mean square of the GRU hidden
  state at every step.
- `depth_levels`: `(B,)` scalar tag of the configured trunk depth.
- `ring_levels`: `(B,)` scalar tag of the configured number of rings.
- `anchor_levels`: `(B,)` scalar tag of the number of anchors.

## Implementation Binding

- Registered model name: `ring_shell_recurrent_boardnet`
- Source implementation file: `src/chess_nn_playground/models/trunk/ring_shell_recurrent_boardnet.py`
- Idea-local wrapper: `ideas/registry/i168_ring_shell_recurrent_boardnet/model.py`

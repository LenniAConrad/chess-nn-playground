# Architecture

`Piece-Plane Gated CNN` is a board-only classifier for the repository
`simple_18` tensor contract. It treats the first 12 planes as semantically
typed piece planes and the final 6 planes as side/state planes instead of
feeding all channels through a single image stem.

## Channel Groups

For `simple_18`, the model uses the known repository channel mapping:

- White piece planes: `P, N, B, R, Q, K`.
- Black piece planes: `p, n, b, r, q, k`.
- State planes: side-to-move, castling rights, and en-passant.

Each group is processed by its own small convolutional stem with shared
`group_width` output size. If a non-`simple_18` channel count is requested, the
implementation switches to deterministic contiguous groups and exposes
`semantic_grouping_known = 0` in the output diagnostics.

## Gates And Fusion

The gate summary is built from safe board-only counts: per-piece occupancy
counts, state-plane means, white-minus-black piece deltas, and grouped totals.
A learned MLP maps this summary to sigmoid gates for the white, black, and
state feature stems.

The gated group features are concatenated, projected with a `1x1` convolution,
and passed through an ordinary residual CNN trunk. The classifier head uses
mean and max global pooling followed by an MLP to emit one puzzle logit for
the puzzle-binary task.

## Ablations

- `ungrouped_stem_matched`: replace the semantic group stems with one matched
  single stem over all input planes.
- `no_gates`: keep semantic group stems but use unit gates.
- `random_channel_groups`: preserve group sizes but assign channels to groups
  using a fixed random permutation.

## Implementation Binding

- Registered model name: `piece_plane_gated_cnn`
- Source implementation file: `src/chess_nn_playground/models/piece_plane_gated_cnn.py`
- Idea-local wrapper: `ideas/i145_piece_plane_gated_cnn/model.py`

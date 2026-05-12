# Architecture

`Sparse Witness-Piece Bottleneck Network` implements the packet's hard occupied-piece rationale bottleneck. The model first validates the board-channel contract, then forces the classifier to see only a fixed-budget subset of occupied piece-squares plus safe global state planes.

## Modules

- `EncodingAdapter` extracts current-board piece planes, safe global planes, a global state vector, and an occupied-square mask. The default `simple_18` binding uses channels `0..11` as current pieces and `12..17` as side/castling/en-passant planes. Non-`simple_18` encodings must provide explicit adapter indices and fail closed otherwise.
- `BoardScorer` scores only current occupied piece-squares with the packet's small scorer: `Conv3x3(12+G, 32) -> GELU -> Conv3x3(32, 32) -> GELU -> Conv1x1(32, 1)`.
- `OccupiedPieceTopKSelector` applies hard top-k selection over valid occupied squares only, selecting `min(K, n_occupied)` witnesses per board. Training uses straight-through Gumbel top-k; evaluation uses deterministic top-k.
- `WitnessGridEncoder` receives only the censored piece planes, the hard binary witness mask, and broadcast global state planes. It uses `Conv3x3 -> GELU`, configurable small residual blocks, global average pooling, and a compact MLP head.
- `SparseWitnessBottleneckNet` composes the adapter, scorer, selector, and witness encoder.

## Forward Contract

Input is a board tensor `[B, C, 8, 8]`. For `simple_18`, the adapter computes:

```text
piece = x[:, 0:12]
global_planes = x[:, 12:18]
global_vec = mean(global_planes, dim=(2, 3))
occupied = clamp(sum(piece, dim=1, keepdim=True), 0, 1)
```

The scorer produces `[B, 1, 8, 8]` raw scores and masks empty squares to a large negative value before selection. The selector returns a hard binary witness mask `[B, 1, 8, 8]`. No continuous selector probabilities are concatenated to the classifier input.

The witness grid is:

```text
witness_piece = piece * mask
witness_grid = concat(witness_piece, mask, broadcast(global_vec))
```

For `simple_18`, this gives `12 + 1 + 6 = 19` input channels to `WitnessGridEncoder`.

With `num_classes: 1`, the repo puzzle-binary trainer receives one logit of shape `[B]` for the idea contract where fine labels `0` and `1` are non-puzzle and fine label `2` is puzzle. With `num_classes: 2`, the same encoder returns logits `[B, 2]` for packet-style two-class experiments.

`forward(x)` returns logits only. `forward_with_mask(x)` is available for tests and reports and returns `(logits, mask, raw_scores)`.

## Implementation Binding

- Registered model name: `sparse_witness_piece_bottleneck_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/sparse_witness_bottleneck.py`
- Idea-local wrapper: `ideas/registry/i038_sparse_witness_piece_bottleneck_network/model.py`

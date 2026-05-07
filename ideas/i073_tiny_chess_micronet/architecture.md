# Architecture

`Tiny Chess MicroNet` implements the packet's tiny learned local field plus fixed
chess sketch bank. The default config targets the `micro_25k` tier: it uses a
`width: 16` hidden field, three micro line blocks, and a 32-unit descriptor head
with about 22k trainable parameters.

## Low-Rank Channel Squeeze

The repository board tensor contract is:

```text
x: (B, 18, 8, 8)
```

The input adapter is a low-rank quantization-friendly projection:

```text
Conv1x1(18 -> squeeze_rank)
ReLU6
Conv1x1(squeeze_rank -> width)
ReLU6
```

For the default `micro_25k` config this is `18 -> 6 -> 16`.

## Micro Line Blocks

Each block applies the packet's residual local-plus-line update:

```text
h = ReLU6(h + alpha * MicroLineBlock(h))
```

`MicroLineBlock` contains:

- depthwise `3x3` local filtering;
- fixed rank/file/diagonal/anti-diagonal line smoothing;
- learned per-channel line gammas for the four directions;
- low-rank `1x1` channel mixing `width -> mix_rank -> width`;
- ReLU6 activations and a learned residual scale initialized to `0.1`.

The line smoothing matrices are registered buffers, so the model does not allocate
rank/file/diagonal masks inside `forward`.

## Chess Sketch Bank

The hidden field is never flattened into a large dense head in the main model.
Instead `ChessSketchBank` emits four descriptor groups:

- global pools: per-channel mean, max, and mean absolute deviation;
- fixed line sketches over rank, file, diagonal, and anti-diagonal directions using
  constant, edge-heavy, center-heavy, side-relative forward, side-relative backward,
  and occupancy-weighted bases;
- own and opponent king-zone pools for 3x3 zone, 5x5 ring, and nearest-edge zone;
- material/state summaries from `simple_18`: piece counts, side to move, castling
  count, en-passant flag, total occupancy, own occupancy, and opponent occupancy.

King-zone decoding uses the `simple_18` king planes. If a board does not contain
exactly one white king or one black king, the king descriptor fails closed to zeros
and reports `malformed_king_count` rather than raising during training.

## Descriptor Head

The head is an INT8-ready low-rank MLP:

```text
descriptor -> Linear(head_hidden) -> ReLU6 -> Linear(num_classes)
```

For `puzzle_binary`, `num_classes: 1`, so forward returns `output["logits"]` with
shape `(B,)`. The output dictionary also includes descriptor energies, learned
group norm fractions for global, line, king-zone, and material groups, parameter
count, FP32 size estimate, and simulated INT8 size estimate.

## Ablation Hooks

Supported `model.ablation` values are `counts_only_mlp`,
`ordinary_tiny_cnn_matched`, `flat_head_same_params`, `no_line_sketch`,
`random_line_basis`, `no_king_zone`, and `no_depthwise_local`.

## Implementation Binding

- Registered model name: `tiny_chess_micronet`.
- Source implementation file: `src/chess_nn_playground/models/tiny_chess_micronet.py`.
- Idea-local wrapper: `ideas/i073_tiny_chess_micronet/model.py`.

# Architecture

`Square-Color Parity Mixer` is a board-only puzzle-binary classifier that makes the dark/light square bipartition an explicit modeling axis.

## Input And Tokenization

- Input is the repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- A compact convolutional square encoder projects the board planes to one learned token per square.
- The 64 square tokens are split into fixed dark-square and light-square sets using `(rank + file) % 2`, producing two 32-token subspaces.

## Piece-Conditioned Parity Gates

The model reads deterministic piece-type occupancy from the first 12 `simple_18` planes and collapses color into six type channels: pawn, knight, bishop, rook, queen, and king. A learned sigmoid gate produces two values for every square:

- `within` controls same-color mixing inside the dark or light subspace.
- `cross` controls opposite-color mixing between the dark and light subspaces.

The gate is initialized with chess priors from the thesis: bishops favor within-color flow, knights favor cross-color flow, and pawns, queens, and kings start with mixed parity behavior. The gates remain trainable.

## Parity Block Mixer

Each mixer layer has the block form

```text
[ A_dark    C_cross ]
[ C_cross^T A_light ]
```

where `A_dark` and `A_light` are learned row-normalized 32-by-32 same-color mixers and `C_cross` is a learned 32-by-32 cross-color mixer. The forward pass computes:

- dark same-color update from `A_dark * dark`
- light same-color update from `A_light * light`
- dark cross update from `C_cross * light`
- light cross update from `C_cross^T * dark`

The piece-conditioned gates modulate the projected same-color and cross-color updates before residual normalization and a per-subspace feed-forward block. This preserves the proposal's dark/light block structure while allowing learned interaction strength by piece type and square.

## Readout

The classifier pools convolutional board summaries, dark/light token summaries, parity sum and difference summaries, and gate/block-energy diagnostics. It returns one puzzle logit for the `puzzle_binary` task. Fine labels 0 and 1 are non-puzzle; fine label 2 is puzzle.

## Diagnostics

The output dictionary includes `logits` plus parity diagnostics such as `within_gate_mean`, `cross_gate_mean`, `bishop_within_gate`, `knight_cross_gate`, `pawn_cross_gate`, `within_block_energy`, `cross_block_energy`, `cross_within_ratio`, and dark/light token energy summaries.

## Implementation Binding

- Registered model name: `square_color_parity_mixer`.
- Source implementation: `src/chess_nn_playground/models/square_color_parity_mixer.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i127_square_color_parity_mixer/model.py`.

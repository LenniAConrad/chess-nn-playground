# Architecture

## Scaffold-Only Implementation Notice

This folder is not a completed bespoke implementation of the architecture described below. `model.py` is a thin `ResearchPacketProbe` wrapper built with `build_research_packet_probe_from_config`, so this idea remains `implementation_kind: shared_probe_variant` and `implementation_status: probe_scaffold_only` until bespoke model code matching this markdown is added.


`Commutative View-Consistency Network` is a board-only multi-view architecture
for the `puzzle_binary` task. It consumes the repo `simple_18` tensor with
shape `(batch, 18, 8, 8)` and emits one puzzle logit with shape `(batch,)`.

## Current-Board Views

- `z_square`: compact convolutional square encoder, mean+max pooled.
- `z_piece`: DeepSets over occupied piece-square tokens.
- `z_line`: MLP over rank, file, diagonal, and anti-diagonal summaries.
- `z_region`: MLP over fixed centre/edge/corner masks plus king-centred rings.
- `z_count`: MLP over material, phase, side-to-move, castling, and en-passant
  summaries.

Each view is projected to latent width `D = 32` by default.

## Low-Rank View Maps

The model learns factorised rank-8 maps between view latents:

```text
A_square_to_line
A_line_to_square
A_square_to_region
A_piece_to_region
A_region_to_count
A_square_to_count
A_count_to_square
A_region_to_piece
```

## Defect Features

The classifier reads direct cross-view residuals and two-step cycle residuals:

```text
z_line   - A_square_to_line(z_square)
z_square - A_line_to_square(z_line)
A_square_to_region(z_square) - A_piece_to_region(z_piece)
A_region_to_count(z_region)  - A_square_to_count(z_square)
z_square - A_count_to_square(z_count)
z_piece  - A_region_to_piece(z_region)
z_square - A_line_to_square(A_square_to_line(z_square))
z_piece  - A_region_to_piece(A_piece_to_region(z_piece))
z_square - A_count_to_square(A_square_to_count(z_square))
```

For each residual, the model computes mean-square, mean-absolute, signed-mean,
maximum-absolute, and cosine-consistency statistics. The head reads projected
defect features, projected view summaries, and the raw defect statistics.

## Outputs

The forward pass returns a dictionary with `logits` shaped `(batch,)`,
aggregate diagnostics (`consistency_energy`, `mean_defect_l1`,
`mean_defect_cosine`), per-view latent norms, and per-defect statistics.

## Implementation Binding

- Registered model name: `commutative_view_consistency_network`.
- Source implementation: `src/chess_nn_playground/models/research_architectures.py`.
- Idea-local wrapper: `ideas/i137_commutative_view_consistency_network/model.py`.

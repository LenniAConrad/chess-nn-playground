# Architecture

`Commutative View-Consistency Network` is a board-only multi-view architecture
for the `puzzle_binary` task. It consumes the repo `simple_18` tensor with
shape `(batch, 18, 8, 8)` and emits one puzzle logit with shape `(batch,)`.

## Implementation Binding

- Registered model name: `commutative_view_consistency_network`.
- Source implementation file: `src/chess_nn_playground/models/trunk/commutative_view_consistency.py`.
- Idea-local wrapper: `ideas/registry/i137_commutative_view_consistency_network/model.py`.
- Mechanism family: `symmetry`.
- Input contract: simple_18 only (18 planes, 8x8); the model raises on other encodings or input-channel counts.
- Output contract: puzzle_binary one-logit (`num_classes = 1`); the model raises if requested otherwise.

## Current-Board Views

Each view is produced by a dedicated encoder that consumes only safe
current-board signals and projects to a shared latent width `D = latent_dim`
(default `32`).

- `z_square` (`_SquareEncoder`): compact convolutional trunk over the
  simple_18 board (depth `depth`, width `channels`, BatchNorm/GroupNorm,
  optional Dropout2d). Pooled by concatenating mean and max over the 8x8 grid
  and projected to `D`.
- `z_piece` (`_PieceDeepSets`): per-square tokens built from the 12-d piece
  one-hot plus 4 coordinate features (rank, file, centre distance, square
  parity); mean-aggregated with an occupancy mask so the operator is
  permutation-invariant in the piece-token set and projected to `D`.
- `z_line` (`_MLPEncoder`): LayerNorm + GELU MLP over 30-d rank, file,
  diagonal, and anti-diagonal occupancy summaries.
- `z_region` (`_MLPEncoder`): LayerNorm + GELU MLP over 8-d centre, edge,
  corner, inner-4, and king-ring (radius 1 / 2) occupancy summaries.
- `z_count` (`_MLPEncoder`): LayerNorm + GELU MLP over 25-d material counts,
  per-piece deltas, side-to-move, castling, en-passant, and material balance.

## Low-Rank View Maps

Eight factorised rank-`map_rank` (default `8`) linear maps connect the view
latents:

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

Each map is implemented as a `Linear(D, rank)` followed by `Linear(rank, D)`
so the operator has factorised rank `≤ rank`. Only the maps themselves carry
parameters; the view registry is non-learnable.

## Defect Features

The model computes six direct cross-view residuals:

```text
z_line   - A_square_to_line(z_square)
z_square - A_line_to_square(z_line)
z_region - A_square_to_region(z_square)
z_count  - A_region_to_count(z_region)
z_square - A_count_to_square(z_count)
z_piece  - A_region_to_piece(z_region)
```

and three two-step cycle residuals:

```text
z_square - A_line_to_square(A_square_to_line(z_square))
z_piece  - A_region_to_piece(A_piece_to_region(z_piece))
z_square - A_count_to_square(A_square_to_count(z_square))
```

For every defect the model computes five scalar statistics: mean-square,
mean-absolute, signed-mean, max-absolute, and cosine consistency between the
target and the predicted latent. These nine defect vectors (`9 × 5 = 45`
statistics) are flattened and concatenated with the five projected view
latents to form the classifier head input.

## Head

`[view_pooled.flatten(1), defect_stats.flatten(1)]` (shape `(B, 5D + 45)`) is
LayerNormed, projected through a GELU MLP, and reduced to one puzzle logit.
The forward pass also returns the per-view latents, per-view RMS norms,
per-defect L2/L1/cosine statistics, `consistency_energy = mean(defect_l2^2)`,
`mean_defect_l1`, `mean_defect_cosine`, the bookkeeping diagnostics
`commutative_view_ablation`, `commutative_view_count`, `mechanism_energy`,
`proposal_profile_strength`, and `proposal_keyword_count`.

The view encoders and the cross-view maps are the load-bearing learnable
modules; the deterministic line/region/count summaries and the coordinate
plane buffer of the piece DeepSets are non-learnable. This makes the model
materially distinct from a generic multi-branch CNN (no defect statistics in
the head), from the kinematic-commutator family (defects here are between
*learned* view-to-view maps rather than deterministic motion operators), and
from the shared `ResearchPacketProbe` scaffold (which has no view encoders).

## Supported ablations

`CommutativeViewConsistencyNetwork.ABLATIONS` enumerates the testable variants:

- `none` — full implementation as described above.
- `views_only_no_defects` — zero out every defect feature so the head reads
  only the five projected view summaries. Tests whether defect features add
  information beyond per-view summaries.
- `single_square_view` — disable the piece/line/region/count encoders by
  zeroing their latents, so the head sees only the square latent. Tests
  whether the multi-view system is needed at all.
- `random_view_maps` — freeze the cross-view maps at deterministic random
  scale-matched values (seeded via `random_map_seed`). Tests whether the
  learned maps add information beyond fixed-scale residual regularizers.
- `count_to_all_only` — restrict every defect path to start from `z_count`
  by zeroing the other view latents before the maps are applied. Tests
  whether the model is collapsing to a material shortcut.
- `shuffled_piece_view` — permute the per-square piece tokens across the
  batch before the piece DeepSets encoder runs. Tests whether the piece
  view contributes real geometry.

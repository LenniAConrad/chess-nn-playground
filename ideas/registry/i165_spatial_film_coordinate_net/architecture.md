# Architecture

`Spatial FiLM Coordinate Net` is a board-only classifier for the
`puzzle_binary` task. It accepts the repo's simple 18-plane current-board
tensor with shape `(B, 18, 8, 8)` and returns one puzzle logit per position.

Coordinate features per square are deterministic and precomputed as a buffer:

```text
c_s = [rank, file, center_distance, edge_distance,
       side_relative_rank, square_color]
```

`rank, file, side_relative_rank` lie in `[-1, 1]`, `center_distance` is the
Chebyshev distance from the centre, `edge_distance = 1 - center_distance`,
and `square_color` is `+1` on light squares and `-1` on dark squares.

The trunk is a stack of `depth` ConvBlocks of width `width`. Each ConvBlock is
``Conv3x3 -> BatchNorm -> GELU -> Dropout``. A coordinate-conditioned FiLM
generator runs in parallel for each layer:

```text
gamma_l(s), beta_l(s) = MLP_l(c_s)            # bounded by tanh
h_l = ConvBlock(h_{l-1})                       # standard CNN block
h_l = gamma_l(s) * h_l + beta_l(s)             # spatial FiLM modulation
```

with bounded modulation

```text
gamma_l = 1 + 0.25 * tanh(raw_gamma_l)
beta_l  = 0.25 * tanh(raw_beta_l)
```

Pooled mean+max features are concatenated and a small MLP head produces one
logit. Coordinate features and the FiLM generators are precomputed once; the
coordinate grid is registered as a buffer rather than rebuilt every forward.

Implemented ablations:

- `coord_planes_only` -- append the coordinate tensor as input planes and
  disable FiLM. Tests whether spatial FiLM beats plain coordinate planes.
- `no_side_relative_coord` -- zero out the side-relative coordinate channel.
- `shared_gamma_only` -- use a per-layer global channel gamma vector and no
  per-square beta. Tests whether spatial modulation is what matters.
- `random_coord_map` -- deterministically permute coordinate assignment to
  squares so coordinate semantics are scrambled.
- `cnn_matched_params` -- disable FiLM and coordinate planes; the trunk is a
  plain matched-parameter CNN baseline.

Diagnostics returned alongside the logit:

- `gamma_maps` and `beta_maps` of shape `(depth, width, 8, 8)`.
- `modulation_magnitudes`: per-layer modulation energy.
- `modulation_center_mean`, `modulation_edge_mean`, `modulation_back_rank_mean`:
  region statistics over the modulation deviation `|gamma - 1| + |beta|`.
- `coordinate_grid` and `trunk_features` for inspection.
- `depth_levels`: scalar tag of the configured depth.

## Implementation Binding

- Registered model name: `spatial_film_coordinate_net`
- Source implementation file: `src/chess_nn_playground/models/trunk/spatial_film_coordinate_net.py`
- Idea-local wrapper: `ideas/registry/i165_spatial_film_coordinate_net/model.py`

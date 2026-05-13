# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/ray_cast_obstacle_pool_head.py`.
- Idea-local wrapper: `ideas/registry/p026_ray_cast_obstacle_pool_head/model.py`.
- Registry key: `ray_cast_obstacle_pool_head`.
- Source primitive: `ideas/research/primitives/external_22_ray_cast_obstacle_pooling_sparse_emit.md`.
- Shared scaffolding: `src/chess_nn_playground/models/primitives/primitive_heads.py`.

## Inputs

The model consumes only the `simple_18` `(B, 18, 8, 8)` current-board
tensor. The per-square features and the blocker mask are both derived
from the 12 piece planes — no `python-chess` call inside the forward
pass.

## Forward path

1. `trunk(board)` returns the standard i193 diagnostics dict.
2. `occupancy_from_simple_18(board)` sums the 12 piece planes to get
   `O in [0, 1]^{B x 8 x 8}`.
3. `self.feature_proj` projects the 12 piece planes to a per-square
   feature stack of width `feature_dim`.
4. `ray_pool(features, occupancy, gamma, max_ray_length, use_occlusion)`
   runs the 8-direction scan. The inner loop is sequential over ray
   length but vectorised over the batch/feature/spatial axes.
5. Mean-pool to `(B, 8, feature_dim)`, flatten, LayerNorm, fuse with
   trunk diagnostics, and emit the additive gated delta.

## `_shift_along_direction` helper

Implements a zero-padded shift of a `(B, C, 8, 8)` tensor by `(row,
file)` steps. Cells coming from outside the board contribute zero, so a
ray that runs off the board simply stops accumulating.

## Cost model

| Component | Approximate cost |
|---|---|
| `feature_proj` 1x1 conv | `B * 12 * feature_dim * 64` MACs |
| Ray scan | `B * 8 * max_ray_length * feature_dim * 64` MACs |
| Fusion MLP | `(8 * feature_dim + 4) → head_hidden_dim → 1` twice |
| Total head FLOPs | < 10% of the i193 trunk forward at defaults |

## Diagnostics surface area

The forward dict adds the following RayPool-specific diagnostics on top
of the standard primitive diagnostics:

- `raypool_active_squares` — total occupancy per sample.
- `raypool_ray_energy` — mean ray-feature L2 across the 8 directions.
- `raypool_max_dir_energy` — peak directional energy per sample.
- `raypool_gamma_mean` — average value of the learned per-direction
  decay (broadcast per sample).

## Deferred work

- Fused CUDA kernel for the 8-direction scan: the Python loop is small
  on 8x8 but would benefit from a single CUDA call for larger boards or
  multi-stage scans.
- Direction-aware attention: replace the mean-pool with a learned
  per-direction attention head conditioned on the trunk diagnostics.

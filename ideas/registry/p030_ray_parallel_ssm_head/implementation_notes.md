# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/ray_parallel_ssm_head.py`.
- Idea-local wrapper: `ideas/registry/p030_ray_parallel_ssm_head/model.py`.
- Registry key: `ray_parallel_ssm_head`.
- Source primitive: `ideas/research/primitives/external_27_ray_parallel_ssm_delta_accumulator_sparse_conv.md`.
- Shared scaffolding: `src/chess_nn_playground/models/primitives/primitive_heads.py`.

## Inputs

The model consumes only the `simple_18` `(B, 18, 8, 8)` current-board
tensor. The per-square features, A, B, and C are all derived from
the 12 piece planes — no `python-chess` call inside the forward
pass.

## Forward path

1. `trunk(board)` returns the standard i193 diagnostics dict.
2. `features = feature_proj(board[:, :12])` produces a per-square
   feature stack of width `feature_dim`.
3. `A_per_dir = sigmoid(A_proj(features.permute(0, 2, 3, 1)))` emits
   per-(square, direction, channel) retention coefficients.
4. `B_per_dir = sigmoid(B_proj(features.permute(0, 2, 3, 1)))` emits
   per-(square, direction, channel) injection coefficients.
5. Both are reshaped to `(B, NUM_DIRECTIONS, F, 8, 8)` via
   `_split_per_direction`.
6. For each direction, iterate `max_ray_length` steps: shift state,
   multiply by `A[..., d]`, add `B[..., d] * features`. Apply the
   per-direction `C[d]` and accumulate into `y_total`.
7. Mean-pool over the board, LayerNorm, concatenate with trunk
   diagnostics, and emit the gated additive delta.

## Cost model

| Component | Approximate cost |
|---|---|
| `feature_proj` | `B * 12 * feature_dim * 64` MACs |
| `A_proj` | `B * feature_dim * (NUM_DIRECTIONS * feature_dim) * 64` MACs |
| `B_proj` | Same as `A_proj` |
| Scan | `B * NUM_DIRECTIONS * max_ray_length * feature_dim * 64` MACs |
| Fusion MLP | `(feature_dim + 4) → head_hidden_dim → 1` twice |

The dominant cost is `A_proj`/`B_proj` (one large linear each); the
scan loop is cheap on 8x8 because each shift is a single tensor copy.

## Diagnostics surface area

The forward dict adds the following Ray-SSM-specific diagnostics on
top of the standard primitive diagnostics:

- `ray_ssm_mean_A` — mean retention coefficient across all (square,
  direction, channel) entries per sample.
- `ray_ssm_mean_B` — mean injection coefficient.
- `ray_ssm_dir_energy_mean` — average per-direction state energy.
- `ray_ssm_dir_energy_max` — peak per-direction state energy.

## Deferred work

- Fused Mamba-style parallel scan (currently sequential — adequate
  for `max_ray_length=7`).
- Per-(square, direction) `C` (currently per-direction only — the
  spec's full form requires an additional 64-row C table per
  direction).
- Cross-channel A and B (currently diagonal — a full A in
  `R^{F x F}` would be richer but quadratically more expensive).

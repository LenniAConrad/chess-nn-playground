# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/incremental_latent_accumulator_head.py`.
- Idea-local wrapper: `ideas/registry/p028_incremental_latent_accumulator_head/model.py`.
- Registry key: `incremental_latent_accumulator_head`.
- Source primitive: `ideas/research/primitives/external_24_incremental_latent_accumulator_directional_scan.md`.
- Shared scaffolding: `src/chess_nn_playground/models/primitives/primitive_heads.py`.

## Inputs

The model consumes only the `simple_18` `(B, 18, 8, 8)` current-board
tensor. The own-king square is read from the simple_18 piece planes
(planes 5 for white king and 11 for black king) + the side-to-move
plane (channel 12) — no `python-chess` call inside the forward pass.

## Forward path

1. `trunk(board)` returns the standard i193 diagnostics dict.
2. `_own_king_square(board)` returns the side-to-move's king square
   index in `[0, 64]` (with 64 = "no king" dummy row).
3. `_compute_accumulators(board)` returns `(h_global, h_king,
   king_idx)`. Both sparse sums use `torch.einsum`.
4. `h_concat = cat([h_global, h_king], dim=-1)`; the `phi` MLP is
   skipped under the `linear_only` ablation.
5. `latent = LayerNorm(phi(h_concat))`.
6. The latent is concatenated with the trunk diagnostics and fed to
   the gate / delta MLPs. The final logit is
   `base_logit + gate * delta`.

## Why no python-chess in forward

The king-anchored accumulator only needs the own-king square index
and the per-(piece-type, square) indicator. Both are directly
available from `simple_18`. The "incremental" interpretation
(O(1) updates per chess move) still holds in the math, even though
we are running a stateless forward pass per position.

## Cost model

| Component | Approximate cost |
|---|---|
| Global embedding | `12 * 64 * global_dim` floats |
| King-anchored embedding | `65 * 12 * 64 * king_dim` floats |
| Global einsum | `O(active_cells * global_dim)` per sample |
| King-anchored einsum | `O(active_cells * king_dim)` per sample |
| `phi` MLP | Two small Linear layers |
| Fusion MLP | `(global_dim + king_dim + 4) → head_hidden_dim → 1` twice |

At default hyper-params the head's parameter count is dominated by
the king-anchored embedding (~50k floats).

## Diagnostics surface area

The forward dict adds the following ILA-specific diagnostics on top
of the standard primitive diagnostics:

- `ila_global_norm` — `||h_global||_2` per sample.
- `ila_king_norm` — `||h_king||_2` per sample.
- `ila_latent_norm` — `||latent||_2` per sample.
- `ila_active_cells` — number of occupied piece cells.
- `ila_king_index` — own-king square index (cast to float for parquet
  serialisation).

## Deferred work

- Replace the explicit king-anchored table with a relative-square
  encoding (saves memory).
- Two-king (own + enemy) anchored variant.
- Precompute parquet column for the indicator sums.

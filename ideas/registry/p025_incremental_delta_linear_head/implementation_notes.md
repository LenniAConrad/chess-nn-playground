# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/incremental_delta_linear_head.py`.
- Idea-local wrapper: `ideas/registry/p025_incremental_delta_linear_head/model.py`.
- Registry key: `incremental_delta_linear_head`.
- Source primitive: `ideas/research/primitives/external_21_incremental_delta_linear_color_involution_adjacency.md`.
- Shared scaffolding: `src/chess_nn_playground/models/primitives/primitive_heads.py`.

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. The IDL embedding is read from the 12 piece planes flattened to
`(B, 12, 64)` — no `python-chess` call inside the forward pass.

## Forward path

1. `trunk(board)` runs the bespoke i193 dual-stream network and returns
   the standard diagnostics dict.
2. `_compute_accumulator(board)` flattens the piece planes and applies
   any ablation-driven shuffle / permutation, then sparse-sums the
   `(12, 64, accumulator_dim)` embedding via `torch.einsum`.
3. The normalised state, the four trunk diagnostics, and a single
   `||S||` scalar are concatenated and passed through the gate / delta
   MLPs.
4. The final logit is `base_logit + gate * delta`, with the usual zero-
   delta / disable-gate / trunk-only switches handled in
   `fuse_with_base_logit`.

## Why no python-chess in forward

The IDL operator is *defined* on the piece-plane indicators only —
piece-type at square *s* under colour *c*. The simple_18 piece planes
already give this directly. There is no need to enumerate legal moves,
look up castling state, or read en-passant; the embedding sum has no
dependence on those rules. This keeps the head wall-clock comparable to
the trunk forward.

## Cost model

| Component | Approximate cost |
|---|---|
| Embedding table | `12 * 64 * accumulator_dim` floats (~36k @ d=48) |
| einsum forward | `O(active_cells * accumulator_dim)` per sample |
| Fusion MLP | `(accumulator_dim + 5) → head_hidden_dim → 1` twice |
| Total head FLOPs | < 5% of the i193 trunk forward at default hyper-params |

## Diagnostics surface area

The forward dict adds the following IDL-specific diagnostics on top of
the i193 trunk + standard primitive diagnostics:

- `idl_accumulator_norm` — `||LayerNorm(S)||_2` per sample.
- `idl_accumulator_state_l2` — pre-norm `||S||_2` per sample.
- `idl_active_cells` — number of occupied piece cells (sanity check).

These are pushed into the per-sample predictions parquet for slice
reporting.

## Deferred work

- Precompute path: the IDL embedding sum is already cheap, but a
  matching precompute parquet column (`p025_accumulator`) would let the
  model run head-only at inference time if/when this primitive is
  promoted into a hybrid.
- IEL color-involution: a follow-on head should tie weights across the
  `(color_swap, vertical_flip)` group instead of computing them
  independently. That belongs in a separate primitive folder.
- HalfKA refinement: the king-anchored variant lives in `p028` (ILA).

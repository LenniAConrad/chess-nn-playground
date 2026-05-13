# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/sparse_legal_move_router_head.py`.
- Idea-local wrapper: `ideas/registry/p027_sparse_legal_move_router_head/model.py`.
- Registry key: `sparse_legal_move_router_head`.
- Source primitive: `ideas/research/primitives/external_23_sparse_legal_move_router_kinematic_state_space.md`.
- Shared scaffolding: `src/chess_nn_playground/models/primitives/primitive_heads.py`.

## Inputs

The model consumes only the `simple_18` `(B, 18, 8, 8)` current-board
tensor. The legal-move adjacency is computed rule-exactly from the
piece planes + the side-to-move plane via the i193 geometry helpers
(`_build_geometry`, `_piece_channel`, `between`, `clear`). No
`python-chess` call inside the forward pass.

## Forward path

1. `trunk(board)` returns the standard i193 diagnostics dict.
2. `_piece_type_per_square(board)` classifies each board square into one
   of 13 classes (12 piece types + empty); the piece-type embedding
   plus the per-square positional embedding produce the token features.
3. `compute_legal_move_adjacency(board)`:
   - Reads `simple_18` piece planes and the side-to-move plane.
   - Reads the i193 `geom_attacks` and `between` tables.
   - For each (piece type, color) combination, accumulates the per-
     source attack mask multiplied by the line-clear indicator for
     sliding pieces.
   - Selects own-color contributions via the side-to-move plane.
4. Q/K/V projections, masked softmax, sum-by-edges to produce the
   routed feature.
5. Mean-pool over squares, LayerNorm, concatenate with trunk
   diagnostics, run the gate / delta MLPs.

## Why no python-chess in forward

The legal-move adjacency we materialise treats *piece-type movement*
as the connectivity (jump piece moves + sliding piece moves with
blocker termination). It does not consult the king-in-check
restriction, en-passant target, castling rights, or 50-move rule. This
captures the SLMR primitive's claim ("information should flow along
legal piece moves") rule-exactly for the purpose of routing while
keeping the helper a pure PyTorch operation.

If a stricter "actually legal under check / pin" mask is needed, the
follow-up production path is a precomputed parquet column:

```text
scripts/data/precompute_primitive_features.py  (TODO)
   reads data/splits/.../{train,val,test}.parquet
   computes the full legal-move adjacency per row via python-chess
   writes a sibling split with column `legal_move_adjacency: list[bool]`.
```

When that script lands the dataset can expose the mask as a batch
tensor and `compute_legal_move_adjacency` becomes a tensor index
instead of a geometry-table accumulation.

## Cost model

| Component | Approximate cost |
|---|---|
| `compute_legal_move_adjacency` | `(B, 64, 64)` einsum + 6 piece-type accumulations |
| Q/K/V matmul | `B * 64 * attn_dim * square_embed_dim` MACs |
| Attention | `B * 64 * 64 * attn_dim` MACs |
| Fusion MLP | `(attn_dim + 4) → head_hidden_dim → 1` twice |

At default attn_dim=32 the attention matmul is the dominant per-step
cost in the head.

## Diagnostics surface area

The forward dict adds the following SLMR-specific diagnostics on top
of the standard primitive diagnostics:

- `slmr_legal_move_edges` — total edges in the adjacency per sample.
- `slmr_active_sources` — number of source squares with at least one
  legal target.
- `slmr_attention_entropy` — normalised attention entropy averaged
  across active sources.
- `slmr_routed_feature_norm` — `||pooled||_2` per sample.

## Deferred work

- A precompute parquet column for the king-safe legal-move adjacency,
  exposed via `data.primitive_feature_columns`.
- A piece-type-aware edge embedding for the attention key.
- Multi-round routing (currently single round).

# Architecture

`Sparse Legal-Move Router Head` (p027) is an additive, gated head over
the i193 trunk. Promoted from the first-ranked proposal of
`ideas/research/primitives/external_23_sparse_legal_move_router_kinematic_state_space.md`.

The model consumes the `simple_18` `(B, 18, 8, 8)` board tensor and
returns one puzzle logit plus the standard primitive diagnostics dict.

## Mechanism

1. **i193 trunk forward**. Emits the canonical diagnostics.
2. **Per-square tokens**. A piece-type lookup table (13 classes — 12
   piece planes + "empty") plus a learned 64-row per-square positional
   embedding form `(B, 64, square_embed_dim)`.
3. **Legal-move adjacency**. `compute_legal_move_adjacency(board)` reads
   the simple_18 piece planes + the side-to-move plane, then aggregates
   the per-piece attack mask under the side-to-move's pieces, honouring
   blockers for sliding pieces. The result is a `(B, 64, 64)` 0/1
   tensor that we cache as `adjacency`.
4. **Masked attention**. Q/K/V projections to `attn_dim` width. The
   attention logits are masked so that non-edges become `-inf` before
   softmax. Sources with no legal target fall back to a self-loop so
   softmax does not NaN, and the corresponding output row is then
   masked to zero by `source_has_target`.
5. **Pool + fuse**. The routed `(B, 64, attn_dim)` tensor is mean-pooled
   to `(B, attn_dim)`, LayerNorm'd, concatenated with the four trunk
   diagnostics, and fed to the gate / delta MLPs.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and any other report-only metadata are *not* consumed. The legal-move
adjacency is constructed rule-exactly from the simple_18 piece planes +
the side-to-move plane only.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Adjacency construction | One `einsum` over `between` + piece-type sums |
| Attention | `O(64 * 64 * attn_dim)` for Q/K/V matmuls + softmax |
| Fusion head | Two MLPs over `(attn_dim + 4)` |

At the default `attn_dim=32` and trunk size, the head is ~20% of the
total per-step cost. The `compute_legal_move_adjacency` helper imports
the i193 geometry tables once at module load and reuses them.

## Deferred external_23 proposals

- **KDS** (Kinematic Deformable Sampling) — a piece-type-conditioned
  convolution offset table. Better placed in the trunk's feature
  builder.
- **ISSC** (Incremental State-Space Cell) — covered by `p028` and
  `p030`.
- **CIF** (Color-Invariant Folding) — an activation-only primitive;
  out of scope for an additive head.
- **BPIP** (Bilinear Piece-Interaction Pooling) — overlaps with the
  DHPE/PPI family (`i245`).

## Implementation Binding

- Registered model name: `sparse_legal_move_router_head`.
- Source implementation: `src/chess_nn_playground/models/primitives/sparse_legal_move_router_head.py`.
- Idea-local wrapper: `ideas/registry/p027_sparse_legal_move_router_head/model.py`.
- Training config: `ideas/registry/p027_sparse_legal_move_router_head/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["sparse_legal_move_router_head"] = build_sparse_legal_move_router_head_from_config`.

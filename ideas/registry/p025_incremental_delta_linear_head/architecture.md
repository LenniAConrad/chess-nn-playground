# Architecture

`Incremental Delta-Linear Accumulator Head` (p025) is an additive, gated
head over the i193 `ExchangeThenKingDualStreamNetwork` trunk. It promotes
the *first-ranked* proposal of
`ideas/research/primitives/external_21_incremental_delta_linear_color_involution_adjacency.md`
— the **Incremental Delta-Linear Operator (IDL)**.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit for the BCE-with-logits
`puzzle_binary` trainer, plus a per-sample diagnostics dict that combines
the i193 trunk diagnostics with the new `primitive_*` and `idl_*` outputs.

## Mechanism

1. **i193 trunk forward**. The bespoke
   `ExchangeThenKingDualStreamNetwork` runs unchanged and emits the
   standard diagnostics dict (`logits` as `base_logit`, `exchange_logit`,
   `king_logit`, `gate`, `gate_logit`, `residual_logit`, `gate_entropy`,
   `stream_disagreement`, `exchange_pool_norm`, `king_pool_norm`,
   `mechanism_energy`, `proposal_profile_strength`,
   `proposal_keyword_count`).

2. **IDL accumulator**. The 12 piece planes are flattened to
   `(B, 12, 64)`. A learned `(12, 64, accumulator_dim)` embedding table
   `E` is sparse-summed via `torch.einsum('bts,tsd->bd', planes, E)` to
   produce the accumulator state `S` of shape `(B, accumulator_dim)`.
   This is the pure linear half of NNUE / HalfKA expressed as a standalone
   operator on top of the i193 trunk.

3. **Fusion**. The normalised state `LayerNorm(S)` is concatenated with
   four stop-gradient i193 diagnostics (`gate`, `gate_entropy`,
   `mechanism_energy`, `stream_disagreement`) and a single scalar `||S||`,
   then passed through two small LayerNorm + GELU MLP heads:

   - `gate_mlp` → scalar gate logit → sigmoid → `primitive_gate`
   - `delta_mlp` → scalar `primitive_delta_raw`

   The final logit is

   ```text
   final_logit = base_logit + primitive_gate * primitive_delta_raw
   ```

4. **Ablations**. See `ablations.md` for the complete table. The primary
   falsifier is `shuffle_squares` (random column permutation of the piece
   planes before the einsum) — if the architecture matches the shuffled
   run on the declared target slice, the per-square factorisation of `E`
   is not load-bearing.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata are
*not* consumed by the model. The IDL embedding is read from the 12 piece
planes of `simple_18` only — that is, from legal-board state.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| IDL sparse sum | One `einsum` of cost O(active_cells * accumulator_dim) |
| Fusion head | Two small MLPs over a `(accumulator_dim + 5)`-d vector |

The IDL accumulator is cheap compared to the trunk: at the default
`accumulator_dim=48`, the embedding table is ~36k parameters and the
einsum is dominated by 32 active piece cells per typical position.

## Deferred external_21 proposals

The research file contains five proposals. We implement IDL only:

- **IEL (Color-Involutive Equivariant Linear)** — a weight-tying
  constraint that flips channels and rotates spatially. This is a trunk
  feature-builder modification, not a head primitive. Tracked as a future
  research direction in `implementation_notes.md`.
- **AGR (Adjacency-Gated Reduction)** — overlaps directly with the
  Sparse Legal-Move Router primitive (`p027`).
- **PPI (Piece-Pair Interaction Kernel)** — overlaps with the DHPE/PPI
  piece-pair family promoted as `i245`.
- **GSL (Gated Spatial-Temporal Lookup)** — already implemented as part
  of the i193 king-stream feature builder.

## Implementation Binding

- Registered model name: `incremental_delta_linear_head`.
- Source implementation: `src/chess_nn_playground/models/primitives/incremental_delta_linear_head.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p025_incremental_delta_linear_head/model.py`.
- Training config: `ideas/registry/p025_incremental_delta_linear_head/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["incremental_delta_linear_head"] = build_incremental_delta_linear_head_from_config`.

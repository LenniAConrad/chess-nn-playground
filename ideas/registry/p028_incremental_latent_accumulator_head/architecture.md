# Architecture

`Incremental Latent Accumulator Head` (p028) is an additive, gated head
over the i193 trunk that promotes the ILA primitive from
`ideas/research/primitives/external_24_incremental_latent_accumulator_directional_scan.md`.

The model consumes the `simple_18` `(B, 18, 8, 8)` board tensor and
returns one puzzle logit plus the standard primitive diagnostics dict.

## Mechanism

1. **i193 trunk forward**. Emits the canonical diagnostics.
2. **Global accumulator**. The 12 piece planes are flattened to
   `(B, 12, 64)` and sparse-summed with a learned
   `(12, 64, global_dim)` table — same shape as `p025` / IDL.
3. **Own-king square**. `_own_king_square(board)` selects the
   side-to-move's king plane (white king plane 5 or black king plane
   11) and reads the argmax over the flattened plane. If the king is
   missing (corrupted data), the "no king" dummy index 64 is used.
4. **King-anchored accumulator**. A learned
   `(65, 12, 64, king_dim)` table is indexed by the own-king square,
   yielding a `(B, 12, 64, king_dim)` per-sample sub-table. The
   piece-plane indicator is then sparse-summed across `(piece-type,
   square)` to give `h_king` of shape `(B, king_dim)`.
5. **Non-linear lift**. `phi_mlp(h_concat)` runs a LayerNorm + GELU +
   Linear stack; `linear_only` ablation skips it and `LayerNorm` is
   still applied so the output magnitude is comparable across modes.
6. **Fusion**. The latent `z` is concatenated with the four trunk
   diagnostics; two MLPs emit the gate logit and the delta.
7. **Additive logit**. `final_logit = base_logit + gate * delta`.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and any other report-only metadata are *not* consumed. The own-king
square is read rule-exactly from the simple_18 piece planes + the
side-to-move plane.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Global accumulator | One `einsum('bts,tsd->bd', ...)` |
| King-anchored accumulator | One index + one `einsum('bts,btsd->bd', ...)` |
| `phi_mlp` + LayerNorm | Two small Linear layers over `(global_dim + king_dim)` |
| Fusion head | Two MLPs over `(global_dim + king_dim + 4)` |

At the defaults (`global_dim=48`, `king_dim=16`) the head is ~50k
floats of state and well under 5% of the trunk forward cost.

## Deferred external_24 proposals

- **LMTG** (Legal-Move Topology Gate) — overlaps with `p027` SLMR.
- **BPDO** (Bit-Population Differentiable Operator) — separate
  counting / logic primitive outside this batch.
- **SEI** (Symmetry-Equivariant Involution) — trunk-level weight-
  tying primitive.
- **DSS** (Directional Stopping Scan) — overlaps with `p029` OARS
  and `p030` Ray-SSM.

## Implementation Binding

- Registered model name: `incremental_latent_accumulator_head`.
- Source implementation: `src/chess_nn_playground/models/primitives/incremental_latent_accumulator_head.py`.
- Idea-local wrapper: `ideas/registry/p028_incremental_latent_accumulator_head/model.py`.
- Training config: `ideas/registry/p028_incremental_latent_accumulator_head/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["incremental_latent_accumulator_head"] = build_incremental_latent_accumulator_head_from_config`.

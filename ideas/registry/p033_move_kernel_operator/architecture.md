# Architecture

`Move-Kernel Operator` (p033, MKO) is an additive, gated head on top of
the i193 `ExchangeThenKingDualStreamNetwork` trunk. The thesis (see
`math_thesis.md`) is that chess-rule geometric structure provides a
natural cross-square weight-sharing pattern that standard Conv2d does not
exploit.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits ``base_logit`` and trunk diagnostics.
2. **Per-square seed feature**. ``X = Linear(13)(piece_planes + stm)``
   shape ``(B, 64, feature_dim)``.
3. **Per-type masks**. Static buffers (T, 64, 64); built once from the
   chess geometry tables and registered in the model state. Six active
   types: knight, rank, file, diag, antidiag, king.
4. **Per-type projection**. ``W_t X`` for each ``t`` (6 Linear passes).
   The output is ``(B, T, 64, feature_dim)``.
5. **Masked aggregation**. ``Y[b, i, d] = sum_t sum_j M_t[i, j] *
   (W_t X)[b, j, d]`` via a single batched einsum.
6. **Pool**. Concatenate own-piece-weighted mean and global mean.
7. **Delta head**. Two-layer MLP -> scalar ``primitive_delta_raw``.
8. **Gate**. MLP over trunk diagnostics + per-type activation magnitude
   summary -> sigmoid ``primitive_gate``.
9. **Output**. ``final_logit = base_logit + primitive_gate *
   primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full MKO architecture (default). |
| `shared_kernel` | Collapse all 6 types to one shared projection. Tests whether move-type weight sharing is load-bearing. |
| `scalar_per_type` | Per-type *scalar* gain instead of a per-type matrix. Tests whether the matrix capacity is load-bearing. |
| `shuffle_features` | In-batch permutation of the seed features. Decouples rule features from position. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations, and
principal variations are **not** consumed by the model. The move-type
masks depend on chess geometry only; piece presence enters only via the
per-square seed feature.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Per-type masks | Static buffer; no per-batch cost |
| Per-type projection | 6 Linear(d, d) passes |
| Aggregation | One batched einsum (T, 64, 64) x (B, T, 64, d) |
| Per-type norm diagnostic | One additional einsum (used to feed the gate) |
| Head / gate | Small MLPs |

At defaults (``feature_dim=24``, T=6, ``head_hidden_dim=64``) the head
adds ~0.5M parameters.

## Implementation Binding

- Registered model name: `move_kernel_operator`.
- Source implementation: `src/chess_nn_playground/models/primitives/move_kernel_operator.py`.
- Geometry helper: `src/chess_nn_playground/models/primitives/legal_move_graph.py`
  (the static reach masks are derived from the same `_Geometry` cache).
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p033_move_kernel_operator/model.py`.
- Training config: `ideas/registry/p033_move_kernel_operator/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["move_kernel_operator"] = build_move_kernel_operator_from_config`.

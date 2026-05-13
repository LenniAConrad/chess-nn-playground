# Architecture

`Dynamic Adjacency-Conditioned Gating` (p032, DAG) is an additive, gated
head on top of the i193 `ExchangeThenKingDualStreamNetwork` trunk. The
thesis (see `math_thesis.md`) is that the legal-move adjacency carries a
move-type structure (rank / file / diagonal / antidiagonal / knight /
king / pawn) that a single shared kernel must average over, while a
per-type kernel can specialise.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits `base_logit` and the standard trunk
   diagnostics (`gate`, `gate_entropy`, `mechanism_energy`,
   `stream_disagreement`, ...).
2. **Legal-move graph**. ``compute_legal_move_graph`` produces ``A`` and
   the per-edge ``move_type`` code. The adjacency is computed inside
   ``torch.no_grad()``.
3. **Per-type decomposition**. For each ``t in T`` (8 active move types,
   see ``ACTIVE_MOVE_TYPES``), build the binary mask
   ``A_t[b, i, j] = A[b, i, j] * 1[move_type[b, i, j] == t]``.
4. **Per-square seed feature**. ``X = Linear(13)(piece_planes + stm)``
   shape ``(B, 64, feature_dim)``.
5. **Per-type projection**. ``W_t @ X`` for each ``t``, giving
   ``(B, T, 64, feature_dim)``.
6. **Masked aggregation**. ``Y[b, i, d] = sum_t sum_j A_t[b, i, j] *
   (W_t X)[b, j, d]`` via a single batched einsum.
7. **Pool**. Concatenate own-piece-weighted mean and global mean.
8. **Delta head**. Two-layer MLP -> scalar ``primitive_delta_raw``.
9. **Gate**. MLP over trunk diagnostics + per-type adjacency degree
   summaries -> sigmoid ``primitive_gate``.
10. **Output**. ``final_logit = base_logit + primitive_gate *
    primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full DAG architecture (default). |
| `single_move_type` | Collapse all types to one shared projection. Tests whether per-type kernel specialisation is load-bearing. |
| `soft_mask` | Replace binary mask with sigmoid(2 * (A - 0.5)). Tests whether the hard-mask commitment is load-bearing. |
| `uniform_adjacency` | Replace adjacency with all-ones (minus identity). Tests whether the rule-derived edges matter at all. |
| `shuffle_adjacency` | In-batch permutation of the legal-move graph. Decouples rule indicators from positions. |
| `zero_delta` | Zero primitive delta. Recovers i193 trunk behavior. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin the gate at 1.0. Tests whether the gate is load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations, and
principal variations are **not** consumed by the model. The legal-move
graph depends only on the simple_18 piece planes, side-to-move plane, and
blocker occupancy.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Legal-move graph | One einsum (blocker resolution) + masked sliding/jump edges |
| Per-type masks | One broadcast + equality + product (no learned weights) |
| Per-type projection | 8 Linear(d, d) passes over (B, 64, d) inputs |
| Aggregation | One batched einsum (B, T, 64, 64) x (B, T, 64, d) |
| Head / gate | Small MLPs |

At defaults (``feature_dim=24``, T=8, ``head_hidden_dim=64``) the head
adds roughly 0.6M parameters on top of the i193 trunk.

## Implementation Binding

- Registered model name: `dynamic_adjacency_gating`.
- Source implementation: `src/chess_nn_playground/models/primitives/dynamic_adjacency_gating.py`.
- Legal-move graph helper: `src/chess_nn_playground/models/primitives/legal_move_graph.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p032_dynamic_adjacency_gating/model.py`.
- Training config: `ideas/registry/p032_dynamic_adjacency_gating/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["dynamic_adjacency_gating"] = build_dynamic_adjacency_gating_from_config`.

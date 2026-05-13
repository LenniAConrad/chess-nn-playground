# Architecture

`Sparse Legal-Move Graph Transition` (p035, SLMGT) is an additive,
gated head on top of the i193 `ExchangeThenKingDualStreamNetwork`
trunk. The thesis (see `math_thesis.md`) is that a learned *joint*
edge function over the legal-move graph captures source-target
interaction features that 1-hop legal-mask attention and per-type
linear projections miss.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits ``base_logit`` and trunk diagnostics.
2. **Legal-move graph**. ``compute_legal_move_graph`` produces ``A``
   (and ``own_piece_mask``, ``degree``). Computed inside
   ``torch.no_grad()``.
3. **Per-square seed feature**. ``X = Linear(13)(piece_planes + stm)``
   shape ``(B, 64, feature_dim)``. Default ``feature_dim = 16``.
4. **Edge function**.

       phi(X_i, X_j) = LayerNorm(ReLU(
           W_self  X_i
         + W_neighbor X_j
         + W_interact (X_i ⊙ X_j)
       ))

   ``W_self / W_neighbor / W_interact`` are Linear(feature_dim,
   edge_hidden_dim) projections. Default ``edge_hidden_dim = 24``.
5. **Mask + mean aggregation**.

       Y[i] = sum_j A[i, j] * phi(X_i, X_j) / max(degree[i], 1).

6. **Aggregator MLP**. ``LayerNorm + Linear + GELU`` to
   ``head_hidden_dim``.
7. **Pool**. Concatenate own-piece-weighted mean and global mean.
8. **Delta head**. Two-layer MLP -> scalar ``primitive_delta_raw``.
9. **Gate**. MLP over trunk diagnostics + degree / edge-norm /
   edge-max summary -> sigmoid ``primitive_gate``.
10. **Output**. ``final_logit = base_logit + primitive_gate *
    primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full SLMGT architecture (default). |
| `separable_phi` | Zero the Hadamard interaction term ``W_interact (X_i ⊙ X_j)``. Tests whether the joint edge function is load-bearing. |
| `uniform_adjacency` | Replace ``A`` with all-ones (minus identity). Tests whether the chess-rule mask matters. |
| `shuffle_adjacency` | In-batch permutation of the legal-move graph. Decouples rule indicators from positions. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations, and
principal variations are **not** consumed by the model. The legal-move
graph depends only on the simple_18 piece planes, side-to-move plane,
and blocker occupancy.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Legal-move graph | One einsum (blocker resolution) + masked sliding/jump edges |
| Edge function | (B, 64, 64, edge_hidden) pair tensor + LayerNorm |
| Aggregation | Mean over neighbours per source square |
| Head / gate | Small MLPs |

At defaults (``feature_dim=16``, ``edge_hidden_dim=24``, B=128) the
pair tensor uses ~50MB of GPU memory. The head adds ~0.4M parameters.

## Implementation Binding

- Registered model name: `sparse_legal_graph_transition`.
- Source implementation: `src/chess_nn_playground/models/primitives/sparse_legal_graph_transition.py`.
- Legal-move graph helper: `src/chess_nn_playground/models/primitives/legal_move_graph.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p035_sparse_legal_graph_transition/model.py`.
- Training config: `ideas/registry/p035_sparse_legal_graph_transition/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["sparse_legal_graph_transition"] = build_sparse_legal_graph_transition_from_config`.

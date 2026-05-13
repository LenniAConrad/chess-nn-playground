# Architecture

`Legal-Move Laplacian Resolvent` (p031, LM-LPP) is an additive, gated head
on top of the existing i193 `ExchangeThenKingDualStreamNetwork` trunk. The
thesis (see `math_thesis.md`) is that a content-dependent resolvent
operator over the legal-move graph captures multi-hop tactical influence
that single-hop masked attention cannot.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit for the BCE-with-logits
`puzzle_binary` trainer, plus a rich per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. The bespoke
   `ExchangeThenKingDualStreamNetwork` runs unchanged and emits
   `base_logit` plus diagnostics (`gate`, `gate_entropy`, `mechanism_energy`,
   `stream_disagreement`, ...).

2. **Legal-move adjacency**. ``compute_legal_move_graph`` produces a
   ``(B, 64, 64)`` float adjacency, ``(B, 64)`` own / enemy piece masks,
   and per-edge `move_type` / `ray_direction` codes. The adjacency is
   computed inside ``torch.no_grad()`` -- edge existence depends on
   discrete chess-rule indicators.

3. **Piece-weighted adjacency**. Multiply each row of the adjacency by
   the learned per-piece weight ``w(piece(i, x))``. The ``uniform_piece_weights``
   ablation skips this step.

4. **Signed Laplacian**. ``L = D - W`` where ``D = diag(row-sum W)``.

5. **Per-square seed features**. Project the 13-d
   ``(piece-existence + side-to-move)`` descriptor of each square to a
   ``feature_dim`` vector. Default ``feature_dim = 32``.

6. **Neumann series**. Compute
   ``Y = sum_{k=0..K} alpha^k * L^k * X`` with ``K = neumann_terms``
   (default 4). The effective ``alpha = alpha_init * tanh(alpha_logit)``
   so ``|alpha| < alpha_init`` always. Mix with ``Theta`` (a linear
   ``d -> d`` map without bias).

7. **Pooling**. Concatenate the own-piece-weighted mean and the global
   mean of ``Y`` along the channel axis, then project through
   ``LayerNorm + Linear + GELU`` to ``head_hidden_dim`` (default 64).

8. **Delta head**. A two-layer MLP turns the pooled summary into a
   scalar ``primitive_delta_raw``.

9. **Gate**. A small MLP over four detached trunk diagnostics plus three
   resolvent spectral summaries (``mean_norm``, ``max_norm``,
   ``degree_mean``) produces a sigmoid ``primitive_gate``. ``disable_gate``
   pins the gate at 1.

10. **Output**. ``final_logit = base_logit + primitive_gate * primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full architecture (default). |
| `k1_gat_rebrand` | Force K=1. Source primitive's named failure mode. If unablated matches, the Neumann expansion is not load-bearing. |
| `uniform_piece_weights` | Drop ``w(piece)``. Tests whether the piece-conditioned weight is load-bearing. |
| `shuffle_adjacency` | In-batch permutation of the legal-move graph. Decouples the rule indicators from the position. |
| `zero_alpha` | ``alpha = 0`` -> ``Y = X * Theta``. Tests whether the propagation is load-bearing at all. |
| `zero_delta` | Zero primitive delta. Recovers the i193 trunk. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin the gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, and principal variations are **not** consumed by the
model. The legal-move adjacency is derived from `simple_18` piece planes,
side-to-move plane, blocker occupancy, and chess-rule geometry only.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Legal-move graph | One einsum (blocker resolution) + masked sliding/jump edges |
| Resolvent | `K` batched 64x64x32 matmuls + final 32-d Theta linear |
| Head / gate | Small MLPs over `feature_dim` and `head_hidden_dim` |

At default sizes (``feature_dim = 32``, ``K = 4``, ``head_hidden_dim = 64``)
the head adds approximately 0.5M parameters on top of the i193 trunk.

## Implementation Binding

- Registered model name: `legal_move_laplacian_resolvent`.
- Source implementation: `src/chess_nn_playground/models/primitives/legal_move_laplacian_resolvent.py`.
- Legal-move graph helper: `src/chess_nn_playground/models/primitives/legal_move_graph.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p031_legal_move_laplacian_resolvent/model.py`.
- Training config: `ideas/registry/p031_legal_move_laplacian_resolvent/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["legal_move_laplacian_resolvent"] = build_legal_move_laplacian_resolvent_from_config`.

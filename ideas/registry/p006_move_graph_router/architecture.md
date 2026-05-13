# Architecture — p006 Move-Graph Router

Move-Graph Router (p006) is an additive, gated head on top of the i193
`ExchangeThenKingDualStreamNetwork` trunk. The trunk is unmodified; the
head runs as a side branch, consumes the simple_18 board and a stop-
gradient view of the trunk's pool feature, and emits a logit delta.

## Mechanism

1. **i193 trunk forward**. The bespoke
   `ExchangeThenKingDualStreamNetwork` runs unchanged and emits
   `logits` (the base logit), the per-stream logits, gate, residual,
   mechanism_energy, and the i193 diagnostics dict.

2. **Per-square token tower**. The simple_18 board `(B, 18, 8, 8)` is
   projected by a small 1x1 Conv → GELU → 1x1 Conv → LayerNorm to
   per-square tokens `(B, 64, d)`. This is cheap and intentional —
   the MGR primitive is about routing, not feature extraction, so the
   trunk continues to do the heavy lifting.

3. **Legal-edge mask**. The legal-move graph is computed
   analytically from the simple_18 piece planes plus the precomputed
   geometric attack/between-square tables. Side-to-move selects own
   pieces; occlusion gates sliding-piece relations; targets occupied by
   own pieces are removed. The result is a sparse `(B, 64, 64)` 0/1
   adjacency treated as `torch.no_grad`.

4. **Gather → MLP → scatter-add**. Per-edge messages
   `φ_θ([x_i, x_j])` are produced by a shared two-layer MLP. The
   `(B, 64, 64, d)` message tensor is masked by the legal-edge
   adjacency and mean-pooled over `j` per source (degree-normalised
   with a 1-floor for empty rows). A global mean over sources then
   yields a `(B, d)` feature.

5. **Delta + gate**. A two-layer MLP collapses the global feature to
   the scalar `primitive_delta_raw`. The gate MLP runs on the i193
   trunk's joint pool (detached) and sigmoids to `primitive_gate`. The
   final logit is

   ```text
   final_logit = base_logit + primitive_gate * primitive_delta_raw
   ```

   The gate is initialised near-closed (`gate_init = -2.0`) so the
   primitive starts as a near no-op and the optimisation can amplify
   it only when the rule-derived routing produces real signal.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, PVs, and report-only metadata are *not* consumed.
The legal-edge mask is derived from the simple_18 piece planes,
side-to-move, and the precomputed geometric tables, which is the same
contract that i193 and the other primitive heads (i246, i248) follow.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Token tower | Two 1x1 convs over 18 input channels |
| Edge MLP | `O(64^2 · 2d)` element-wise gather + shared MLP, masked |
| Heads | Two small MLPs on `(B, d)` and `(B, trunk_pool_dim)` |

Dense-mixer worst case is the 64x64 edge tensor; the mask collapses
the effective FLOP count to roughly `|E_b| / 4096 ≈ 0.7%` of dense.

## Implementation Binding

- Registered model name: `move_graph_router`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/move_graph_router.py`.
- Shared rule-graph helpers:
  `src/chess_nn_playground/models/primitives/rule_graph_features.py`.
- Trunk source:
  `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper:
  `ideas/registry/p006_move_graph_router/model.py`.
- Training config:
  `ideas/registry/p006_move_graph_router/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["move_graph_router"] = build_move_graph_router_from_config`.

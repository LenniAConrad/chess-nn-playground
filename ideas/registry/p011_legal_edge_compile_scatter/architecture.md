# Architecture — p011 Legal-Edge Compile Scatter

p011 is an additive, gated head on top of the i193 trunk; the trunk is
unmodified.

## Mechanism

1. **i193 trunk forward** — unchanged.
2. **Per-square token tower** — Conv1x1 → GELU → Conv1x1 → LayerNorm
   produces `(B, 64, d)`.
3. **Typed legal-move adjacency** —
   `_compute_typed_legal_edges(board)` returns `(B, 6, 64, 64)` 0/1
   adjacency from the simple_18 board (reused from p009).
4. **Per-edge σ-gate** — for each piece type, a small two-layer MLP
   maps `[x_src, x_dst]` to a logit; sigmoid times the typed
   adjacency masks off non-edge entries. Gates are
   `(B, 6, 64, 64)`.
5. **Per-type message projection** — six `nn.Linear(d, m)` modules
   (or one shared under `shared_type_weight`) produce the source-
   feature contribution.
6. **Gate-weighted scatter** — `bmm(gate.T, projected)` per type,
   degree-normalised by the typed in-gate-mass, summed over types,
   LayerNorm.
7. **Delta + gate** — mean over squares, two-layer MLP to scalar
   delta. Gate MLP runs on detached i193 trunk pool.

```text
final_logit = base_logit + primitive_gate * primitive_delta_raw
```

## Inputs not used

CRTK metadata, source labels, verification flags, Stockfish scores, PVs,
and report-only metadata are not consumed.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass |
| Token tower | Two Conv1x1 |
| Typed adjacency | `O(6 · 64^2)` mask construction (shared with p009) |
| Edge gate MLP | 6 × `(B, 64, 64, 2d) → 1` MLP |
| Per-type linear + bmm | 6 × `(B, 64, m) → (B, 64, 64) → (B, 64, m)` |
| Heads | Two small MLPs |

## Implementation Binding

- Registered model name: `legal_edge_compile_scatter`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/legal_edge_compile_scatter.py`.
- Shared typed-legal-edge compiler:
  `chess_nn_playground.models.primitives.legal_move_graph_delta._compute_typed_legal_edges`.
- Shared rule-graph helpers:
  `src/chess_nn_playground/models/primitives/rule_graph_features.py`.
- Idea-local wrapper:
  `ideas/registry/p011_legal_edge_compile_scatter/model.py`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.

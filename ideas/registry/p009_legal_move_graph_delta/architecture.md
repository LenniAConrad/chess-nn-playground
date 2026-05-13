# Architecture — p009 Legal-Move-Graph Convolution

p009 is an additive, gated head on top of the i193 trunk; the trunk is
unmodified.

## Mechanism

1. **i193 trunk forward** — unchanged.
2. **Per-square token tower** — Conv1x1 → GELU → Conv1x1 → LayerNorm
   projects `(B, 18, 8, 8)` to `(B, 64, d)`.
3. **Typed legal-move adjacency** —
   `_compute_typed_legal_edges(board)` produces `(B, 6, 64, 64)`
   piece-type-stratified bitboards from the simple_18 piece planes,
   side-to-move, occupancy, and the geometric tables. Stop-gradient.
4. **Per-type linear projections** — six `nn.Linear(d, m)` modules
   produce the source-feature contribution per piece type. Under the
   `shared_weight` ablation a single linear is broadcast across types.
5. **Aggregate** — `bmm(edges_per_type, projected)` reduced over
   source axis; degree-normalised by `edges_per_type.sum(-1)` with a
   1-floor; summed over piece types.
6. **LayerNorm + pool + delta + gate** — LayerNorm over the
   `(B, 64, m)` messages, mean-pool over squares, two small MLPs.

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
| Typed adjacency | `O(6 · 64^2)` mask construction |
| Linears + bmm | `(B*6, 64, 64) @ (B*6, 64, m)` |
| Heads | Two small MLPs |

## Implementation Binding

- Registered model name: `legal_move_graph_delta`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/legal_move_graph_delta.py`
  (also exports `_compute_typed_legal_edges`, reused by p011).
- Shared rule-graph helpers:
  `src/chess_nn_playground/models/primitives/rule_graph_features.py`.
- Idea-local wrapper:
  `ideas/registry/p009_legal_move_graph_delta/model.py`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.

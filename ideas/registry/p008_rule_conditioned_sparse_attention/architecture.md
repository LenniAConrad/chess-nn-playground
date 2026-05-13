# Architecture — p008 Rule-Conditioned Sparse Attention (MobScan)

p008 is an additive, gated head on top of the i193 trunk; the trunk is
unmodified.

## Mechanism

1. **i193 trunk forward** — unchanged.
2. **Per-square token tower** — small Conv1x1 → GELU → Conv1x1 →
   LayerNorm projects `(B, 18, 8, 8)` to `(B, 64, d)`.
3. **Legal-edge adjacency** — `compute_legal_move_graph(board)` returns
   `(B, 64, 64)` 0/1 adjacency from the simple_18 piece planes,
   side-to-move, occupancy, and the precomputed geometric tables.
   Stop-gradient.
4. **Input-conditioned gates** — three linear projections of the token
   embedding produce `A_s, B_s, C_s ∈ [0, 1]^{state_dim}` via sigmoid.
5. **Weight-tied recurrence** — for `t = 1..num_iterations` (default 3):

   ```text
   inbound = norm_edges.T  @  h_{t-1}    # (B, 64, state_dim)
   h_t    = A * inbound + B * input_proj(tokens)
   ```

   where `norm_edges = edges / in_degree`. `B * input_proj(tokens)`
   is reinjected each step so the operator does not erase its own
   input signal across iterations.
6. **Read-out** — `y_s = C_s ⊙ h^T_s`. Mean over squares yields
   `(B, state_dim)`.
7. **Delta + gate** — two small MLPs collapse to the scalar delta and
   sigmoid gate; the gate runs on the detached i193 trunk joint pool.

```text
final_logit = base_logit + primitive_gate * primitive_delta_raw
```

## Inputs not used

CRTK metadata, source labels, verification flags, Stockfish scores, PVs,
and report-only metadata are not consumed. The adjacency uses only the
simple_18 board.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass |
| Token tower | Two Conv1x1 over 18 inputs |
| Gate projections | Three Linear(token_embed_dim, state_dim) |
| Recurrence | `num_iterations` (B, 64, 64) bmm |
| Heads | Two small MLPs |

`num_iterations = 3` keeps the head well within 1.2x i193 wall-clock at
default widths.

## Implementation Binding

- Registered model name: `rule_conditioned_sparse_attention`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/rule_conditioned_sparse_attention.py`.
- Shared rule-graph helpers:
  `src/chess_nn_playground/models/primitives/rule_graph_features.py`.
- Idea-local wrapper:
  `ideas/registry/p008_rule_conditioned_sparse_attention/model.py`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.

# Implementation Notes — p009 Legal-Move-Graph Convolution

- Idea-local wrapper:
  `ideas/registry/p009_legal_move_graph_delta/model.py`.
- Registry key: `legal_move_graph_delta`.
- `implementation_kind: bespoke_model`.
- The per-type adjacency is built by `_compute_typed_legal_edges` in
  `chess_nn_playground.models.primitives.legal_move_graph_delta`. The
  helper is also used by p011 to avoid duplicating chess-rule code.
- Adjacency built inside `torch.no_grad()`; trunk pool detached before
  the gate MLP.
- The six per-type linears are stacked into a single batched matmul
  `(B*6, 64, 64) @ (B*6, 64, m)` for throughput.
- Diagnostics include per-type message norms
  (`lmgconv_msg_norm_{P,N,B,R,Q,K}`) — useful for the `shared_weight`
  ablation report.
- Tests at `tests/test_legal_move_graph_delta.py` cover registry,
  forward shapes, gradient flow, the ablations, edge-count and per-
  type-norm sanity, and rejection of non-simple_18 inputs.

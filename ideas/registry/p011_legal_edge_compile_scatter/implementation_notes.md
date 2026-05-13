# Implementation Notes — p011 Legal-Edge Compile Scatter

- Idea-local wrapper:
  `ideas/registry/p011_legal_edge_compile_scatter/model.py`.
- Registry key: `legal_edge_compile_scatter`.
- `implementation_kind: bespoke_model`.
- The typed legal-edge compiler is shared with p009 via
  `chess_nn_playground.models.primitives.legal_move_graph_delta._compute_typed_legal_edges`.
- Adjacency built inside `torch.no_grad()`. The σ-gate MLP receives
  the gradient and is the only edge-feature path that participates in
  the autograd graph.
- Per-edge σ-gate is computed by six per-type 2-layer MLPs over the
  `(B, 64, 64, 2d)` concat tensor; the result is multiplied
  element-wise with the typed adjacency so off-edge entries stay 0.
- Per-destination normalisation uses the gate-mass `Σ_i g_{r,i,j}`
  with a small `1e-3` floor; this matches the file's "ragged-tensor
  reduction in fp32" mitigation for the mass-vanishing failure mode.
- Diagnostics include `lecs_gate_mean` (mean σ over valid edges) and
  per-type message norms; both are useful for the `no_edge_gate`
  ablation report.
- Tests at `tests/test_legal_edge_compile_scatter.py` cover registry,
  forward shapes, gradient flow, the ablations, edge / gate sanity,
  and rejection of non-simple_18 inputs.

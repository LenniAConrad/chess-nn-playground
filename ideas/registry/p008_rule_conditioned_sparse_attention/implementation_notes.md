# Implementation Notes — p008 Rule-Conditioned Sparse Attention (MobScan)

- Idea-local wrapper:
  `ideas/registry/p008_rule_conditioned_sparse_attention/model.py`.
- Registry key: `rule_conditioned_sparse_attention`.
- `implementation_kind: bespoke_model`.
- Shared rule-graph helpers
  (`src/chess_nn_playground/models/primitives/rule_graph_features.py`)
  provide `compute_legal_move_graph` and the underlying geometric
  tables. The adjacency is built inside `torch.no_grad()`.
- The recurrence is an eager-mode unroll for `num_iterations` steps.
  A custom fused scan kernel is the production path; eager mode is
  sufficient for the scout-scale falsifier and keeps the trainer
  surface minimal.
- Gate `gate_init = -2.0` keeps the primitive near-closed at init so
  the trunk dominates early in training.
- The recurrence reinjects `B * input_proj(tokens)` every step so the
  input signal is not erased across iterations.
- Diagnostics returned: `primitive_delta`, `primitive_gate`,
  `mobscan_edge_count`, `mobscan_state_norm`,
  `mobscan_gate_A_mean`, `mobscan_gate_B_mean`, `mobscan_gate_C_mean`,
  plus i193 trunk diagnostics. The gate means are useful for the
  `untied_state` ablation report — if `A_mean` collapses to 0 or 1
  the selective behaviour is dead.
- Tests at `tests/test_rule_conditioned_sparse_attention.py` cover
  registry, forward shapes, gradient flow, the ablations, edge-count
  sanity, and rejection of non-simple_18 inputs.

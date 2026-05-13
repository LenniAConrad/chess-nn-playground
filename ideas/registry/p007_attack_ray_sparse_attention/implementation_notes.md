# Implementation Notes — p007 Attack-Ray Sparse Attention

- The model is wrapped through
  `ideas/registry/p007_attack_ray_sparse_attention/model.py`, which
  calls
  `chess_nn_playground.models.primitives.attack_ray_sparse_attention.build_attack_ray_sparse_attention_from_config`.
- Registry key: `attack_ray_sparse_attention`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["attack_ray_sparse_attention"] = build_attack_ray_sparse_attention_from_config`.
- `implementation_kind: bespoke_model`. The head wires a bespoke 9-slot
  sparse attention over a rule-derived first-blocker index on top of
  the existing i193 trunk.
- Shared geometric tables (`ray_step_target`, `ray_step_count`,
  `ray_step_valid`) are owned by
  `chess_nn_playground.models.primitives.rule_graph_features` and
  reused across p006-p011.
- `first_blocker_indices` runs inside `torch.no_grad()`; the trunk
  joint pool feature is `.detach()`-ed before the gate MLP.
- Forward returns a dict with `primitive_delta`, `primitive_gate`,
  `arsa_blocker_count`, `arsa_attention_entropy`, `arsa_self_weight`,
  and the i193 trunk diagnostics. Trainer surfaces these in
  `predictions_<split>.parquet` for slice analysis.
- Tests at `tests/test_attack_ray_sparse_attention.py` cover registry,
  forward shapes, gradient flow, the ablations, blocker-count sanity,
  and rejection of non-simple_18 inputs.

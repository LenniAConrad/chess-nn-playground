# Implementation Notes — p010 Ray-Occlusion Semiring Scan

- Idea-local wrapper:
  `ideas/registry/p010_ray_occlusion_semiring_scan/model.py`.
- Registry key: `ray_occlusion_semiring_scan`.
- `implementation_kind: bespoke_model`.
- Ray geometry tables (`ray_step_target`, `ray_step_valid`,
  `ray_step_count`) and `compute_ray_transmittance` live in
  `chess_nn_playground.models.primitives.rule_graph_features`. The
  geometry tables are content-independent and computed once at module
  import.
- Transmittance is computed inside `torch.no_grad()`; the trunk joint
  pool is `.detach()`-ed before the gate MLP.
- Log-domain prefix products with `(1 - O).clamp(eps, 1)` keep the
  scan numerically stable for highly-occupied positions.
- Per-step decay is one learned scalar per direction (8 parameters).
  Decay values are surfaced as `ros_step_decay_mean` diagnostic.
- Tests at `tests/test_ray_occlusion_semiring_scan.py` cover registry,
  forward shapes, gradient flow, the ablations, transmittance bounds,
  and rejection of non-simple_18 inputs.

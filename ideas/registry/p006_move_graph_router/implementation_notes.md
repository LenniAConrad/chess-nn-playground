# Implementation Notes — p006 Move-Graph Router

- The model is wrapped through
  `ideas/registry/p006_move_graph_router/model.py`, which is a thin
  builder that calls
  `chess_nn_playground.models.primitives.move_graph_router.build_move_graph_router_from_config`.
- Registry key: `move_graph_router`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["move_graph_router"] = build_move_graph_router_from_config`.
- `implementation_kind` is `bespoke_model` because the head wires a
  bespoke gather-scatter MLP over a rule-derived adjacency on top of
  the existing i193 trunk; it is not a `ResearchPacketProbe` wrapper.
- Shared rule-graph helpers
  (`src/chess_nn_playground/models/primitives/rule_graph_features.py`)
  carry the geometric tables (`geom_attacks`, `between`, `ray_step_*`)
  that p006-p011 all consume. The tables are computed once at module
  import and registered as module-global; each model copies them onto
  its working device on demand.
- The legal-edge mask is built inside `torch.no_grad()` so the autograd
  graph never sees the discrete adjacency. The trunk's joint pool
  feature is `.detach()`-ed before the gate MLP for the same reason.
- The forward returns a dict containing the i193 trunk diagnostics
  plus the primitive-specific keys (`primitive_delta`, `primitive_gate`,
  `mgr_edge_count`, `mgr_mean_routed_norm`, `mgr_pooled_norm`); the
  trainer surfaces these in `predictions_<split>.parquet` for slice
  analysis.
- Tests live at `tests/test_move_graph_router.py` and cover registry
  membership, forward shapes, gradient flow, edge-count sanity, the
  zero-delta / disable-gate / trunk-only ablations, and rejection of
  non-simple_18 input.
- Config validation: `scripts/validate_training_config.py --static
  ideas/registry/p006_move_graph_router/config.yaml`.
- Smoke training: see `trainer_notes.md` for the scout-scale command.
- Audits: the primitive participates in the standard
  `scripts/ideas/audit_implementation_kinds.py --check` and
  `scripts/ideas/audit_architecture_conformance.py --check` (when the
  audit harnesses are run by CI/operator); `implementation_kind` is
  `bespoke_model` and `mechanism_family` is `legal_routing`.

# Implementation Notes

- Bespoke model code: `src/chess_nn_playground/models/differentiable_chess_fact_lattice.py`.
- Idea-local wrapper: `ideas/i086_differentiable_chess_fact_lattice/model.py` exposes
  `build_model_from_config(config)` and delegates to
  `build_differentiable_chess_fact_lattice_from_config`.
- Registry key: `differentiable_chess_fact_lattice` is registered directly in
  `MODEL_BUILDERS` in `src/chess_nn_playground/models/registry.py` and is
  excluded from `RESEARCH_PACKET_MODEL_NAMES`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0857_tuesday_new_york_diff_ai.md`.
- Input contract: simple_18 current board tensor only;
  CRTK / source / engine metadata is reporting-only and is never used as
  model input.
- Output contract: `(batch,)` BCE logits plus an output dictionary of
  abstract-interpretation diagnostics. The trainer keeps the standard
  puzzle_binary contract.
- Ablation switches (`use_intervals`, `use_meet_channels`,
  `use_ray_transfer`, `use_king_zone`, and the `variant: "pool_control"`
  baseline) are exposed in
  `build_differentiable_chess_fact_lattice_from_config`.

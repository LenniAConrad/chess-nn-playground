# Implementation Notes

- Central code: `src/chess_nn_playground/models/loop_frustration_curvature_network.py`.
- Idea-local wrapper: `ideas/i080_loop_frustration_curvature_network/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_loop_frustration_curvature_network_from_config`.
- Registered model name: `loop_frustration_curvature_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0729_tuesday_new_york_frustration_curvature.md`.
- `build_loop_bank()` produces the static graph buffers
  (`edge_i`, `edge_j`, `edge_type`, `loop_edge_ids`, `loop_edge_mask`,
  `loop_vertex_ids`, `loop_vertex_mask`) with deterministic
  `M = 210`, `L = 520`, `Lmax = Vmax = 12`, matching the packet's
  section-8 geometry.
- The model is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
- Forward returns a dict whose `"logits"` is `(B,)` for the
  repository `puzzle_binary` BCE-with-logits trainer; all other
  entries are diagnostics for prediction artefacts.

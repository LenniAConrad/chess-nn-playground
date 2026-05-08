# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/fisher_geodesic_tension.py`.
- Idea-local wrapper: `ideas/i083_fisher_geodesic_tension_network/model.py` calls
  `build_fisher_geodesic_tension_network_from_config(config["model"])`.
- Registry key: `fisher_geodesic_tension_network`
  (`chess_nn_playground.models.registry.MODEL_BUILDERS`).
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0755_tuesday_new_york_fisher_geodesic.md`.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input. CRTK metadata is
  reporting-only.
- Fisher-Rao geometry (Bhattacharyya coefficient, geodesic excess, and
  the optional spherical hinge angle) is computed in float32 with a
  softmax simplex floor `(1 - 64 * eps) p + eps` to avoid `arccos`
  saturation under mixed-precision training.

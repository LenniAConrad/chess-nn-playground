# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/chess_hypercut_polynomial.py`.
- Idea-local wrapper: `ideas/i082_chess_hypercut_polynomial_network/model.py` calls
  `build_chess_hypercut_polynomial_network_from_config(config["model"])`.
- Registry key: `chess_hypercut_polynomial_network`
  (`chess_nn_playground.models.registry.MODEL_BUILDERS`).
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0733_tuesday_new_york_hypercut_poly.md`.
- This is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input. CRTK metadata is reporting-only.
- The deterministic chess-rule hyperedge builder caches `HyperedgeBatch`
  tensors per board (LRU-style with a soft cap) so repeated positions reuse
  the same hypergraph between epochs.

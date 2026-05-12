# Implementation Notes

- Bespoke model: `src/chess_nn_playground/models/tactical_radius_filtration.py`
  (`TacticalRadiusFiltrationClassifier`,
  `TacticalRadiusGraphBuilder`,
  `build_tactical_radius_filtration_from_config`).
- Registry key: `tactical_radius_filtration`.
- Idea-local wrapper: `ideas/all_ideas/registry/i087_tactical_radius_filtration/model.py`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0857_tuesday_new_york_tactical_radius.md`.
- Board-only: the model consumes the `simple_18` tensor; engine, verification,
  source, and CRTK metadata stay reporting-only.
- Graph construction is deterministic Boolean adjacency over chess
  pseudo-legal contacts; no engine search, no legal best-move generation, no
  policy/value targets.

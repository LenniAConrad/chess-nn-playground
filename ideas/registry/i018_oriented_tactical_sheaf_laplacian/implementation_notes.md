# Implementation Notes

- Central code: `src/chess_nn_playground/models/oriented_tactical_sheaf.py` (`OrientedTacticalSheafNet`, `BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`, `SheafDiffusionBlock`, `TriadDefectPool`).
- Idea-local wrapper: `ideas/registry/i018_oriented_tactical_sheaf_laplacian/model.py` (`build_model_from_config`).
- Registry key: `oriented_tactical_sheaf_laplacian`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0254_tuesday_local_oriented_tactical_sheaf.md`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.

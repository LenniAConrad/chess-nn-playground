# Implementation Notes

- Source implementation: `src/chess_nn_playground/models/local_neighborhood_geometry_network.py`.
- Idea-local wrapper: `ideas/i124_local_neighborhood_geometry_network/model.py`.
- Registry key: `local_neighborhood_geometry_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.
- Batch candidate: `Local Neighborhood Geometry Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Uses V = 6 deterministic, board-only views applied through a single shared encoder; all geometry diagnostics are derived from the resulting V x D embedding cloud.

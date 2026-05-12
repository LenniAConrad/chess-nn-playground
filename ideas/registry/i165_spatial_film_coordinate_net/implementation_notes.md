# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/spatial_film_coordinate_net.py`.
- Idea-local wrapper: `ideas/registry/i165_spatial_film_coordinate_net/model.py`.
- Registry key: `spatial_film_coordinate_net`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.
- Batch candidate: `Spatial FiLM Coordinate Net`.
- This is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input.
- Coordinate features are precomputed as a non-persistent buffer and reused at
  every forward pass, not rebuilt.

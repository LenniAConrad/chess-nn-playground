# Implementation Notes

- Central code: `src/chess_nn_playground/models/differentiable_bitboard_boolean_network.py`.
- Idea-local wrapper: `ideas/i132_differentiable_bitboard_boolean_network/model.py`.
- Registry key: `differentiable_bitboard_boolean_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.
- Batch candidate: `Differentiable Bitboard Boolean Network`.
- Shift table: reuses the deterministic `build_shift_maps` /
  `SHIFT_NAMES` helpers from `bitboard_shift_algebra` so the chess-shape
  shift geometry stays consistent with idea i069.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.

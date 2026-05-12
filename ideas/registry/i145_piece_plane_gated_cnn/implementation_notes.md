# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/piece_plane_gated_cnn.py`.
- Registry key: `piece_plane_gated_cnn`.
- Idea wrapper: `ideas/registry/i145_piece_plane_gated_cnn/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.
- Batch candidate: `Piece-Plane Gated CNN`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The `simple_18` mapping is treated as known: planes `0:6` are white pieces, `6:12` are black pieces, and `12:18` are side/state planes.

# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/line_piece_crossbar_network.py`.
- Registry key: `line_piece_crossbar_network`.
- Idea-local wrapper: `ideas/registry/i171_line_piece_crossbar_network/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.
- Batch candidate: `Line-Piece Crossbar Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The 64x46 piece-line incidence matrix is registered as a non-persistent buffer so it travels with the model on `to(device)` calls but is not saved to checkpoints.

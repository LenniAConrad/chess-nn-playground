# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/piece_conditioned_hypernetwork_cnn.py`.
- Registry key: `piece_conditioned_hypernetwork_cnn`.
- Idea-local wrapper: `ideas/registry/i114_piece_conditioned_hypernetwork_cnn/model.py` calls
  `build_piece_conditioned_hypernetwork_cnn_from_config`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.
- Batch candidate: `Piece-Conditioned Hypernetwork CNN`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Per-sample depthwise convolution is implemented with grouped `conv2d`
  using `groups = B * C`, which keeps the entire block fully
  vectorized on GPU.

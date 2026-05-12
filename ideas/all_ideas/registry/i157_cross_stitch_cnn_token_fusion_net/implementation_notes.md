# Implementation Notes

- Central code: `src/chess_nn_playground/models/cross_stitch_cnn_token_fusion_net.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i157_cross_stitch_cnn_token_fusion_net/model.py`.
- Registry key: `cross_stitch_cnn_token_fusion_net`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.
- Batch candidate: `Cross-Stitch CNN-Token Fusion Net`.
- Architecture is board-only and consumes no engine, verification,
  source, or CRTK metadata as input.
- Reuses `Simple18PieceTokenExtractor` from
  `chess_nn_playground/models/piece_token_cnn_hybrid.py` to produce up
  to `max_piece_tokens` occupied-piece tokens for the token branch.
- Cross-stitch units start as the identity (``A = I``); the model
  begins as the parent late-fusion architecture and the off-diagonal
  entries learn the amount of board <-> token transfer.

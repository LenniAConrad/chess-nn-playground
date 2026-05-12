# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/piece_token_cnn_hybrid.py`
  (defines `Simple18PieceTokenExtractor`, `BoardCNNTrunk`,
  `PieceTokenMixer`, `CNNTokenFusionHead`, `PieceTokenCNNHybrid`,
  and `build_piece_token_cnn_hybrid_from_config`).
- Idea-local wrapper:
  `ideas/registry/i065_piece_token_cnn_hybrid/model.py` exports
  `build_model_from_config` which defaults `num_classes` to 1 and
  delegates to `build_piece_token_cnn_hybrid_from_config`.
- Registry key: `piece_token_cnn_hybrid` (registered explicitly in
  `src/chess_nn_playground/models/registry.py`; removed from
  `RESEARCH_PACKET_MODEL_NAMES` so it no longer auto-binds to the
  shared `ResearchPacketProbe` scaffold).
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2109_friday_shanghai_piece_token_cnn_hybrid.md`.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.

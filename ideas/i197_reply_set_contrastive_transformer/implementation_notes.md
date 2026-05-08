# Implementation Notes

- Central code: `src/chess_nn_playground/models/reply_set_contrastive_transformer.py`.
- Registry key: `reply_set_contrastive_transformer`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Reply-Set Contrastive Transformer`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input. CRTK/source metadata stays reporting-only.
- The idea-local `model.py` is a thin wrapper around
  `build_reply_set_contrastive_transformer_from_config`, no longer a
  `ResearchPacketProbe` scaffold.

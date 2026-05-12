# Implementation Notes

- Central code: `src/chess_nn_playground/models/invertible_board_coupling_network.py`
  (class `InvertibleBoardCouplingNetwork`, builder
  `build_invertible_board_coupling_network_from_config`).
- Idea-local wrapper: `ideas/registry/i122_invertible_board_coupling_network/model.py`
  exposes `build_model_from_config` for the trainer.
- Registry key: `invertible_board_coupling_network` (registered in
  `src/chess_nn_playground/models/registry.py`).
- The model name is excluded from `RESEARCH_PACKET_MODEL_NAMES` so the
  implementation-kind audit detects this folder as `bespoke_model`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.
- Batch candidate: `Invertible Board Coupling Network`.
- This is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input.

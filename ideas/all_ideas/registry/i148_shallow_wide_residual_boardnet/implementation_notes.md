# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/shallow_wide_residual_boardnet.py`
  (`ShallowWideResidualBoardNet`,
  `build_shallow_wide_residual_boardnet_from_config`).
- Idea-local wrapper: `ideas/all_ideas/registry/i148_shallow_wide_residual_boardnet/model.py`
  delegates to `build_shallow_wide_residual_boardnet_from_config`.
- Registry key: `shallow_wide_residual_boardnet` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.
- Batch candidate: `Shallow Wide Residual BoardNet`.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.  Coordinate planes
  are computed from board geometry and the optional count head only
  reads per-channel sums of the simple_18 input.
- The shared `ResearchPacketProbe` scaffold is no longer used; the
  registry entry has been removed from `RESEARCH_PACKET_MODEL_NAMES`.

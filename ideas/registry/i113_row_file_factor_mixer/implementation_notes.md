# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/row_file_factor_mixer.py`.
- Idea-local wrapper: `ideas/registry/i113_row_file_factor_mixer/model.py` calls
  `build_row_file_factor_mixer_from_config`. There is no
  `ResearchPacketProbe` in the path.
- Registry key: `row_file_factor_mixer`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.
- Batch candidate: `Row-File Factor Mixer`.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.

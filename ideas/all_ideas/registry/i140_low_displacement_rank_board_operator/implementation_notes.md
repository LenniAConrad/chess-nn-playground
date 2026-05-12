# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/low_displacement_rank_board_operator.py`.
- Registry key: `low_displacement_rank_board_operator` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Idea-local wrapper:
  `ideas/all_ideas/registry/i140_low_displacement_rank_board_operator/model.py` calls
  `build_low_displacement_rank_board_operator_from_config`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.
- Batch candidate: `Low-Displacement-Rank Board Operator`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The 64x64 operator is materialised explicitly per layer so the per-component
  response statistics (`T_rank`, `T_file`, `H_diag`, `H_anti`, `U V^T`) and
  the displacement residual norm `||A - Z A Z^T||_F` are exact, not
  approximate.

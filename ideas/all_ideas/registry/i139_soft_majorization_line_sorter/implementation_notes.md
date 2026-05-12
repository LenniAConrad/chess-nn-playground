# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/soft_majorization_line_sorter.py`.
- Builder: `build_soft_majorization_line_sorter_from_config` (called from `ideas/all_ideas/registry/i139_soft_majorization_line_sorter/model.py`).
- Registry key: `soft_majorization_line_sorter` (registered in `src/chess_nn_playground/models/registry.py`).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.
- Batch candidate: `Soft Majorization Line Sorter`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The line index buffer (46 lines x 8 padded slots) and the validity mask
  are cached on the module via `register_buffer` so the line geometry
  ships with the model state dict.

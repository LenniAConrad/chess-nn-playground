# Implementation Notes

- Bespoke model: `src/chess_nn_playground/models/bounded_board_hinge_logic.py`.
- Idea-local wrapper: `ideas/i089_bounded_board_hinge_logic/model.py`.
- Registry key: `bounded_board_hinge_logic`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0859_tuesday_new_york_bounded_hinge.md`.
- The model is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
- Geometry tensors (king-zone masks, between-square clearance, the
  per-piece pseudo-legal attack masks, knight and pawn step masks) are
  precomputed once at module construction and reused for every batch.
- The formula library in `BoundedFormulaEvaluator` is fixed: 24 unary,
  96 binary, and 48 king-zone formulas. Only the predicate-bank
  concept/role mixture weights, the `exists` temperature, the PSL
  per-rule positive and negative weights, the head bias, and the head
  temperature are trainable.
- Formula evaluation is chunked along the formula axis to bound peak
  activation memory; the chunk size is exposed as `formula_chunk_size`
  in `config.yaml`.

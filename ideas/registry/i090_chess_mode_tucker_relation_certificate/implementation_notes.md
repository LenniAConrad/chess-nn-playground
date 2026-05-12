# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/trunk/chess_mode_tucker_relation_certificate.py`.
- Idea-local wrapper: `ideas/registry/i090_chess_mode_tucker_relation_certificate/model.py`.
- Registry key: `chess_mode_tucker_relation_certificate`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-28_0900_tuesday_new_york_relation_tucker.md`.
- The model is intentionally board-only and does not consume engine,
  verification, source, principal-variation, mate-score, best-move, or CRTK
  metadata as input.
- The same-parameter control `FlatProjectedMLPControl` lives in the same
  source file and shares the channel lift, GroupNorm, and fixed relation
  tensor with the main model. It can be used in ablation experiments without
  any additional dependencies.
- `fine_label_diagnostic_3x2` returns the `(3, 2)` count and row-rate matrices
  required by the source packet for fine-label reporting.

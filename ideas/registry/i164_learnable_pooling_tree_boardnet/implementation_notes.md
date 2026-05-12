# Implementation Notes

- Central code: `src/chess_nn_playground/models/learnable_pooling_tree_boardnet.py`.
- Registry key: `learnable_pooling_tree_boardnet`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.
- Batch candidate: `Learnable Pooling Tree BoardNet`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The implementation is a coordinate-aware CNN square encoder followed by three
  learnable `2 x 2` pooling nodes (squares -> cells -> quadrants -> root), an
  optional FiLM-style top-down broadcast pass, and a pooled multi-level binary
  classifier.

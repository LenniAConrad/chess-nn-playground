# Implementation Notes

- Central code: `src/chess_nn_playground/models/ring_shell_recurrent_boardnet.py`.
- Registry key: `ring_shell_recurrent_boardnet`.
- Idea-local wrapper: `ideas/registry/i168_ring_shell_recurrent_boardnet/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.
- Batch candidate: `Ring-Shell Recurrent BoardNet`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The two dynamic anchors use a soft centroid of the white-king / black-king planes; if a king plane is empty (e.g. malformed input), the centroid falls back to the geometric center so ring construction stays well defined.
- `num_rings >= 2` is required so the GRU has a non-trivial radial sequence to process; the default of `8` covers any anchor-square pair on an 8x8 board.

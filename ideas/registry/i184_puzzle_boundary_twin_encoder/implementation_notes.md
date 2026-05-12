# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/puzzle_boundary_twin_encoder.py`.
- Idea-local wrapper: `ideas/registry/i184_puzzle_boundary_twin_encoder/model.py`.
- Registry key: `puzzle_boundary_twin_encoder`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.
- Batch candidate: `Puzzle Boundary Twin Encoder`.
- The architecture is intentionally board-only and does not consume
  engine, verification, source, or CRTK metadata as input. The
  trainer can read group ids from the dataset to mine in-batch
  (puzzle, near, random) triples but the model itself does not.
- Pair-margin objective: the trainer adds
  `relu(m_near - boundary_score(puzzle) + boundary_score(near))` and
  optionally
  `relu(m_random_surface - boundary_score(near) + boundary_score(random))`
  on top of the BCE term, reading `boundary_score` directly from the
  forward output.

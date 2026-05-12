# Implementation Notes

- Central code: `src/chess_nn_playground/models/blocker_pin_lattice.py`.
- Registry key: `blocker_pin_lattice_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Blocker-Pin Lattice Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The implementation enumerates fixed slider rays, extracts ordered blocker
  sequences, builds the four lattice states from the packet, and runs a learned
  gated scan over blocker/target tokens before pooling ray-lattice diagnostics
  into the puzzle classifier.
- Supported ablations are `unordered_blockers`, `no_remove_states`, and
  `only_rank_file`.

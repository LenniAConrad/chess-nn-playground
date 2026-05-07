# Math Thesis

Blocker-Pin Lattice Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `5`.

Working thesis: Line tactics are not only about pieces sharing ranks, files, or diagonals. They depend on ordered blockers and pin constraints. A line can be almost tactical, but one blocker order or one unpinned defender changes everything.

Implementation thesis: represent each side-to-move slider ray as a short
latent program over blocker order. The current state, remove-first state,
remove-second state, and side-swap diagnostic expose whether a line tactic is
caused by a specific blocker order or by a pinned defender shielding a valuable
target.

# Math Thesis

Blocker-Pin Lattice Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `5`.

Working thesis: Line tactics are not only about pieces sharing ranks, files, or diagonals. They depend on ordered blockers and pin constraints. A line can be almost tactical, but one blocker order or one unpinned defender changes everything.

This registered implementation tests the thesis through the `logic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

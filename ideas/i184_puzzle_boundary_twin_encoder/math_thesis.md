# Math Thesis

Puzzle Boundary Twin Encoder

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `8`.

Working thesis: The hardest part is the boundary between verified puzzles and verified near-puzzles. Learn that boundary directly with a twin encoder and margin objective.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

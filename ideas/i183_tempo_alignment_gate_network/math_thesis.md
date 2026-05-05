# Math Thesis

Tempo-Alignment Gate Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `7`.

Working thesis: Many near-puzzles are tactical-looking for the wrong side or require a tempo that the side to move does not have. The model should explicitly gate static tactical danger by side-to-move tempo.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

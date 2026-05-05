# Math Thesis

Source-Invariant Puzzle Bottleneck

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `11`.

Working thesis: The dataset has three source groups. A model may accidentally learn source artifacts instead of puzzle structure. This architecture tries to preserve puzzle signal while removing source identity from the main representation.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

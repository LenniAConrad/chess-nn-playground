# Math Thesis

Rank-File Memory Grid Net

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `8`.

Working thesis: Maintain learned memory vectors for each rank and each file. Squares write into their rank/file memories, then rank/file memories write back to squares. This gives global rank/file communication without axial convolutions, line solves, or attention.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

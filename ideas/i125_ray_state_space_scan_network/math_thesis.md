# Math Thesis

Ray State-Space Scan Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `1`.

Working thesis: Chess line motifs often require long-range context, but all-square attention and dynamic attack graphs are not the only way to get it. A state-space scan can process every rank, file, diagonal, and anti-diagonal as a short sequence with shared continuous re...

This registered implementation tests the thesis through the `grammar` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

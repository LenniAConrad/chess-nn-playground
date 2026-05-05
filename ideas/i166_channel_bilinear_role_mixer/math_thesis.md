# Math Thesis

Channel-Bilinear Role Mixer

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `5`.

Working thesis: Ordinary heads pool channels additively. A low-rank bilinear head can explicitly model pairwise interactions between role summaries, such as own-heavy-piece features with opponent-king-zone features, without building square-pair tensors or local product con...

This registered implementation tests the thesis through the `grammar` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

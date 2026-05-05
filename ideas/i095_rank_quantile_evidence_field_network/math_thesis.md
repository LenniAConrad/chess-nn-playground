# Math Thesis

Rank-Quantile Evidence Field Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.

Batch candidate rank: `4`.

Working thesis: Puzzle-likeness may be driven by extreme sparse evidence fields rather than average board evidence. Differentiable rank and quantile pooling can test this while still allowing the classifier to see the full board, unlike a sparse witness mask.

This registered implementation tests the thesis through the `information` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

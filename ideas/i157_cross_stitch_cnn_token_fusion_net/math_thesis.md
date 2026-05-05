# Math Thesis

Cross-Stitch CNN-Token Fusion Net

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.

Batch candidate rank: `3`.

Working thesis: Late fusion between a CNN branch and a piece-token branch may be too weak. A cross-stitch network can let the branches exchange information at multiple depths through learned linear mixing, while still keeping the model practical.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

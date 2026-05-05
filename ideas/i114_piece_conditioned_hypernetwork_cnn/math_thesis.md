# Math Thesis

Piece-Conditioned Hypernetwork CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `2`.

Working thesis: The best local filters may depend on material and piece inventory. A lightweight hypernetwork can condition CNN channel gates or depthwise kernels on safe current-board summaries, adapting the feature extractor without using engine metadata.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

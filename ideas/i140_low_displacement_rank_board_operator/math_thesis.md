# Math Thesis

Low-Displacement-Rank Board Operator

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.

Batch candidate rank: `4`.

Working thesis: Global square mixing can be parameterized by structured matrices instead of dense attention or convolutions. A low-displacement-rank operator over the flattened board can express long-range interactions with Toeplitz/Hankel-like structure and few parameters.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

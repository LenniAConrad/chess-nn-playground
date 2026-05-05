# Math Thesis

Row-File Factor Mixer

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `1`.

Working thesis: Chess boards have two privileged axes: ranks and files. A model can exploit this without a full Transformer by factorizing board processing into rank mixers, file mixers, and piece-channel mixers, then recombining them with bilinear interactions.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

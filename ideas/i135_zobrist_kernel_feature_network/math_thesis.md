# Math Thesis

Zobrist Kernel Feature Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.

Batch candidate rank: `5`.

Working thesis: Zobrist hashing gives chess a compact random fingerprint of piece-square occupancy. A neural model can use many fixed Zobrist-style random feature maps as a cheap kernel approximation, then learn a small classifier over stable board fingerprints.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

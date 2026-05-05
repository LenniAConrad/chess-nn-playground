# Math Thesis

Vector-Quantized Motif Codebook Net

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.

Batch candidate rank: `5`.

Working thesis: Force local board features to pass through a learned discrete codebook. The classifier reads code usage, spatial code maps, and quantized features. This tests whether a compact inventory of board motifs is useful for puzzle-likeness.

This registered implementation tests the thesis through the `sparse` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

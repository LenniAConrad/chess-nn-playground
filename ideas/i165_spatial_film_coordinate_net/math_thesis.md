# Math Thesis

Spatial FiLM Coordinate Net

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `4`.

Working thesis: Appending coordinate planes may be too weak. Instead, generate per-square affine modulation parameters from deterministic coordinate features and side-relative coordinates, then modulate CNN features at multiple depths.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

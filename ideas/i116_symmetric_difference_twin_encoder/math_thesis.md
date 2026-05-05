# Math Thesis

Symmetric Difference Twin Encoder

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `4`.

Working thesis: Safe deterministic board transforms should preserve some evidence and change other evidence. Instead of enforcing invariance, compare the original and transformed board latents by symmetric difference features.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

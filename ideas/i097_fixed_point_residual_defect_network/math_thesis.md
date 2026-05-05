# Math Thesis

Fixed-Point Residual Defect Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `1`.

Working thesis: Puzzle-like positions may be harder for a learned board-state operator to equilibrate. Instead of classifying only the final latent, classify from the residual defects of an unrolled update process:

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

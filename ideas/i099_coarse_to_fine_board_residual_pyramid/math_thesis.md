# Math Thesis

Coarse-to-Fine Board Residual Pyramid

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `3`.

Working thesis: A puzzle-like position may be present in details not explained by coarse board summaries. Build a residual pyramid over the board: classify from what remains after each scale's coarse reconstruction explains the finer scale.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

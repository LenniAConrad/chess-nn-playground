# Math Thesis

Baseline Logit Residual Adapter

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `2`.

Working thesis: The existing simple CNN likely has systematic errors. A small residual adapter can test what information remains after the baseline logit and latent representation are known:

This registered implementation tests the thesis through the `grammar` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

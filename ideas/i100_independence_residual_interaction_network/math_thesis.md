# Math Thesis

Independence Residual Interaction Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `4`.

Working thesis: Some puzzle-like signals may be interactions that remain after subtracting a simple independence explanation of board occupancy. Instead of modeling all piece-square interactions directly, compute signed residuals:

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

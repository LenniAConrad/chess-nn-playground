# Math Thesis

Cross-Scale Attention Residual Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.

Batch candidate rank: `3`.

Working thesis: Puzzle-like evidence may appear when fine-square attention cannot be predicted from coarse board context. This model computes attention from fine tokens to coarse tokens, reconstructs expected fine attention, and classifies from the residual attention map.

This registered implementation tests the thesis through the `graph` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

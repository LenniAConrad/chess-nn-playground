# Math Thesis

Material-Phase Low-Rank Adapter Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `6`.

Working thesis: Chess positions vary greatly by material phase. Instead of one encoder for every position, condition low-rank adapter weights on material summaries while keeping a shared backbone. The architecture tests whether small material-conditioned rank updates impro...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

# Math Thesis

Stripe-Selective Mixer CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.

Batch candidate rank: `4`.

Working thesis: A practical line-aware CNN may be enough to beat the current BT4 while staying simpler than Schur-Ray. Instead of ordinary `3x3` convolutions only, mix along chess stripes:

This registered implementation tests the thesis through the `grammar` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

# Math Thesis

Square-Color Parity Mixer

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `3`.

Working thesis: The chessboard is naturally bipartite by square color. Bishops stay on one color, knights alternate color, kings and queens mix colors locally, and pawn captures switch files and square color. A neural model can explicitly split dark/light square subspaces...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

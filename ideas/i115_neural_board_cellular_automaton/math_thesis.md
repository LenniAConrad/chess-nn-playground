# Math Thesis

Neural Board Cellular Automaton

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `3`.

Working thesis: Some board patterns may be recognized by repeated local relaxation. A neural cellular automaton applies the same local update rule for several steps and classifies from the evolving board state and update energy.

This registered implementation tests the thesis through the `grammar` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

# Math Thesis

Soft Majorization Line Sorter

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.

Batch candidate rank: `3`.

Working thesis: On a tactical line, the exact order and dominance of pieces often matters more than a bag of line pieces. Instead of a ray automaton or line language model, compute differentiable sorted salience profiles along ranks/files/diagonals and classify from majori...

This registered implementation tests the thesis through the `grammar` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

# Math Thesis

Morphological Threat Field Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.

Batch candidate rank: `3`.

Working thesis: CNNs learn filters, but chess tactics often have shape operations: expand a king danger zone, close gaps in a pawn shield, erode escape squares, and detect thin corridors. Differentiable mathematical morphology gives an architecture that explicitly processe...

This registered implementation tests the thesis through the `logic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

# Math Thesis

Auxiliary Reconstruction BoardNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `3`.

Working thesis: A classifier trunk may discard board detail too early. Add a lightweight decoder that reconstructs safe current-board planes from the latent feature map, using reconstruction only as an auxiliary training loss. The classifier still sees no future or engine...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

# Math Thesis

Learnable Pooling Tree BoardNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `3`.

Working thesis: Instead of pooling the whole board at once or using an FPN, build a fixed hierarchy over the `8 x 8` board: squares become `2 x 2` cells, cells become quadrants, quadrants become a board root. Each tree node has a small learned aggregator and passes feature...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

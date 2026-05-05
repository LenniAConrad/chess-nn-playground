# Math Thesis

Shallow Wide Residual BoardNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `6`.

Working thesis: On an `8 x 8` board, depth may be less useful than width and a good head. A shallow wide residual CNN can test whether the benchmark wants broad feature extraction rather than long convolutional stacks.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

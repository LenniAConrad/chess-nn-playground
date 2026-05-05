# Math Thesis

ConvNeXt BoardNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `1`.

Working thesis: Use a small ConvNeXt-style architecture adapted to `8 x 8` chess boards: depthwise spatial mixing, inverted channel MLPs, residual scaling, coordinate planes, and a strong global pooling head.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

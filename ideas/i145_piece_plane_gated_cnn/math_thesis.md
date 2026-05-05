# Math Thesis

Piece-Plane Gated CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `3`.

Working thesis: The `simple_18` channels are not arbitrary image channels. A plain CNN can respect this by first processing semantically related channel groups, then using learned gates to mix piece types and colors.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

# Math Thesis

Board FPN CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `2`.

Working thesis: Chess positions often need both exact square detail and coarse whole-board phase. A plain feature-pyramid network can process the board at `8 x 8`, `4 x 4`, and `2 x 2` resolutions, then fuse the maps back into a single classifier.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

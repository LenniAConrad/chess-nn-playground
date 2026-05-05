# Math Thesis

Ring-Shell Recurrent BoardNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `7`.

Working thesis: Important chess context often radiates from anchors: kings, center squares, edges, and promotion zones. Summarize the board in fixed rings/shells around these anchors and process the shells with a small recurrent model.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

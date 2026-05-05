# Math Thesis

Multi-Order Board Scan Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.

Batch candidate rank: `2`.

Working thesis: A chess board can be read as several short sequences. Different scan orders expose different dependencies: rank-major order, file-major order, diagonal order, spiral-from-king order, and center-out order. A shared sequence model over fixed board orders can...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

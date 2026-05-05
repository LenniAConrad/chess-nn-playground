# Math Thesis

Early-Exit Cascade BoardNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `2`.

Working thesis: Some positions may be easy and should not need a heavy model, while ambiguous near-puzzles need deeper computation. Build a cascade with several classifier exits and train it to produce useful early predictions plus a final refined prediction.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

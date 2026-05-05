# Math Thesis

Axial Rank-File ConvNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `1`.

Working thesis: Use ordinary convolutions, but factor long-range board mixing into alternating `8`-length rank and file convolutions. This gives every square access to same-rank and same-file context cheaply while preserving an ordinary CNN training path.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

# Math Thesis

Legal-Constraint Projection Residual Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.

Batch candidate rank: `4`.

Working thesis: Even when the input board is legal, a learned latent explanation of "why this is puzzle-like" may produce soft piece/square beliefs that violate basic legal-board constraints. Projecting those beliefs back onto a soft legal-board constraint set and reading...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

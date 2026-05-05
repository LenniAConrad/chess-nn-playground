# Math Thesis

Invertible Board Coupling Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.

Batch candidate rank: `4`.

Working thesis: Standard encoders can discard information early, which makes it hard to know whether a model learned legitimate current-board structure or fragile shortcuts. A reversible board encoder preserves information by construction and classifies from latent distort...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

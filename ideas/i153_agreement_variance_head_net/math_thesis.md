# Math Thesis

Agreement-Variance Head Net

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `5`.

Working thesis: Use one shared trunk and several cheap heads trained on the same label. Classify from the mean logits, and log head variance as an uncertainty diagnostic. This is a lightweight alternative to full ensembles.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

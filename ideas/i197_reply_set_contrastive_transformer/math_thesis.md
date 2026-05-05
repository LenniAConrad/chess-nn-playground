# Math Thesis

Reply-Set Contrastive Transformer

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `12`.

Working thesis: A puzzle position should embed differently from its plausible reply positions. A near-puzzle may remain close to one or more safe replies. Use contrastive learning over current position and pseudo-reply positions.

This registered implementation tests the thesis through the `graph` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

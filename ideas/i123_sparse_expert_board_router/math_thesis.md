# Math Thesis

Sparse Expert Board Router

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.

Batch candidate rank: `5`.

Working thesis: Chess positions are heterogeneous. Endgames, king attacks, pawn races, blocked centers, and material imbalances may need different feature extractors. A sparse mixture of small board experts can route positions to specialized encoders without requiring a gi...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

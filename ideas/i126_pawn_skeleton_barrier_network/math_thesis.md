# Math Thesis

Pawn Skeleton Barrier Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `2`.

Working thesis: Pawn structure is a slow, chess-specific skeleton that shapes king safety, open lines, promotion lanes, and tactical vulnerability. A model can compute deterministic pawn barrier and distance fields from the current board, then learn how these fields condit...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

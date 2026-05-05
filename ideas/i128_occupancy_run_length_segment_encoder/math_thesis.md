# Math Thesis

Occupancy Run-Length Segment Encoder

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `4`.

Working thesis: Sliding tactics depend on contiguous empty and occupied segments along ranks, files, and diagonals. Instead of parsing full piece-token ray strings, encode run-length segment summaries: empty run lengths, blocker positions, endpoint piece types, and segment...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

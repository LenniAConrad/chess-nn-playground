# Math Thesis

Zobrist Kernel Feature Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.

Batch candidate rank: `5`.

Working thesis: Zobrist hashing gives chess a compact random fingerprint of piece-square occupancy. A neural model can use many fixed Zobrist-style random feature maps as a cheap kernel approximation, then learn a small classifier over stable board fingerprints.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

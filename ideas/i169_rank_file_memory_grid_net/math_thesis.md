# Math Thesis

Rank-File Memory Grid Net

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `8`.

Working thesis: Maintain learned memory vectors for each rank and each file. Squares write into their rank/file memories, then rank/file memories write back to squares. This gives global rank/file communication without axial convolutions, line solves, or attention.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

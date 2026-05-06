# Math Thesis

Material-Phase Low-Rank Adapter Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `6`.

Working thesis: Chess positions vary greatly by material phase. Instead of one encoder for every position, condition low-rank adapter weights on material summaries while keeping a shared backbone. The architecture tests whether small material-conditioned rank updates impro...

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

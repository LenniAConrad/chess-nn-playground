# Math Thesis

Adapter-Sandwich Residual CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `6`.

Working thesis: Instead of building a much larger new backbone, insert small bottleneck adapters before and after ordinary residual blocks. This tests whether parameter-efficient adapters can improve the existing CNN family while leaving most of the architecture conventional.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

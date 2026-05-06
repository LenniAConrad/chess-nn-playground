# Math Thesis

Channel-Bilinear Role Mixer

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `5`.

Working thesis: Ordinary heads pool channels additively. A low-rank bilinear head can explicitly model pairwise interactions between role summaries, such as own-heavy-piece features with opponent-king-zone features, without building square-pair tensors or local product con...

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

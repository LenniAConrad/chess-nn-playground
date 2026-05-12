# Math Thesis

Axial Rank-File ConvNet

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `1`.

Working thesis: Use ordinary convolutions, but factor long-range board mixing into alternating `8`-length rank and file convolutions. This gives every square access to same-rank and same-file context cheaply while preserving an ordinary CNN training path.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

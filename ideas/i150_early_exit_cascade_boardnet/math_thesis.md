# Math Thesis

Early-Exit Cascade BoardNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `2`.

Working thesis: Some positions may be easy and should not need a heavy model, while ambiguous near-puzzles need deeper computation. Build a cascade with several classifier exits and train it to produce useful early predictions plus a final refined prediction.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

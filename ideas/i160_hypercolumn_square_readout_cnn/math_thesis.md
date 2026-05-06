# Math Thesis

Hypercolumn Square Readout CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.

Batch candidate rank: `6`.

Working thesis: Intermediate CNN layers may detect different chess cues: early local piece contacts, middle motifs, and later global context. A hypercolumn readout gathers per-square features from every depth and classifies from square-level evidence maps plus global pooling.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

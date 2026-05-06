# Math Thesis

Agreement-Variance Head Net

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `5`.

Working thesis: Use one shared trunk and several cheap heads trained on the same label. Classify from the mean logits, and log head variance as an uncertainty diagnostic. This is a lightweight alternative to full ensembles.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

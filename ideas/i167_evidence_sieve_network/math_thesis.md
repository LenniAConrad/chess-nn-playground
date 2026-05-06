# Math Thesis

Evidence Sieve Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `6`.

Working thesis: Instead of refining logits, the model can refine features by repeatedly filtering them through learned evidence sieves. Each sieve stage produces a soft mask over channels and squares, passes selected evidence onward, and leaves a diagnostic trail.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

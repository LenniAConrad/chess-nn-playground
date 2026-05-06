# Math Thesis

Iterative Logit Refinement CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `4`.

Working thesis: Instead of producing a single logit vector at the end, let a model make an initial prediction and then apply several learned correction steps from shared board features. The model tests whether puzzle evidence is better accumulated as staged corrections.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

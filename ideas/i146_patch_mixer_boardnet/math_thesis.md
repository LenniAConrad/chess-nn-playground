# Math Thesis

Patch Mixer BoardNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `4`.

Working thesis: Use a plain MLP-Mixer-style model over `2 x 2` chess patches. This is a simple non-attention alternative to square-token models: mix information across board patches with MLPs, then mix channels with MLPs.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

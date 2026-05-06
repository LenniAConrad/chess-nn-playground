# Math Thesis

Piece-Plane Gated CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `3`.

Working thesis: The `simple_18` channels are not arbitrary image channels. A plain CNN can respect this by first processing semantically related channel groups, then using learned gates to mix piece types and colors.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

# Math Thesis

Specialist-Head CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `5`.

Working thesis: A plain shared CNN trunk can feed several small specialist heads: king-zone head, center-control head, material/phase head, and global board head. A learned fusion layer combines their logits/features. This tests specialization without a full mixture-of-exp...

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

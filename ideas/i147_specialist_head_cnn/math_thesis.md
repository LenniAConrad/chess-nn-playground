# Math Thesis

Specialist-Head CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `5`.

Working thesis: A plain shared CNN trunk can feed several small specialist heads: king-zone head, center-control head, material/phase head, and global board head. A learned fusion layer combines their logits/features. This tests specialization without a full mixture-of-exp...

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

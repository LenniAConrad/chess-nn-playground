# Math Thesis

Patch Mixer BoardNet

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `4`.

Working thesis: Use a plain MLP-Mixer-style model over `2 x 2` chess patches. This is a simple non-attention alternative to square-token models: mix information across board patches with MLPs, then mix channels with MLPs.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

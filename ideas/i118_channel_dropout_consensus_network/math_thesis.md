# Math Thesis

Channel Dropout Consensus Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `6`.

Working thesis: The classifier should not depend too heavily on one piece channel or artifact. Train several shared encoders on deterministic channel-dropped views and classify from consensus and disagreement.

This registered implementation tests the thesis through the `robustness` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

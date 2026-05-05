# Math Thesis

Counterfactual Defender Dropout Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `4`.

Working thesis: If a near-puzzle is only superficially tactical, randomly removing defenders or attackers may not reveal a sharp causal structure. If a true puzzle hinges on overloaded defenders, pinning, or one critical escape square, dropout interventions should produce...

This registered implementation tests the thesis through the `robustness` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

# Math Thesis

Attention Disagreement Residual Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.

Batch candidate rank: `2`.

Working thesis: Near-puzzle and puzzle-like positions may contain competing interpretations. Independent attention query families should disagree more on ambiguous or tactically dense boards. The classifier uses the residual disagreement among attention maps as evidence.

This registered implementation tests the thesis through the `graph` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.

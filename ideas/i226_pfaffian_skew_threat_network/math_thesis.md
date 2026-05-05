# Math Thesis

Pfaffian Skew Threat Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1525_tuesday_local_pfaffian_skew_threat.md`.

Working thesis: Builds a skew-symmetric chess operator K and uses its Pfaffian pf(K) (signed perfect-matching enumerator) plus sub-Pfaffian fingerprints; orientation cancellation discriminates puzzle vs near-puzzle at matched ||K||_F.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.

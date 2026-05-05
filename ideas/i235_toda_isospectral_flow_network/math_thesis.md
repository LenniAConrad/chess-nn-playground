# Math Thesis

Toda Isospectral Flow Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1620_tuesday_local_toda_isospectral_flow.md`.

Working thesis: Builds a tridiagonal symmetric chess operator L and evolves it via the Toda Lax flow dot L = [L, B(L)]; isospectral evolution preserves spectrum but sorts diagonal exponentially. Sorting rate, Manakov integrals, and slowest-decaying off-diagonal residues classify puzzle-likeness.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.

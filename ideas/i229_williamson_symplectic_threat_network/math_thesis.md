# Math Thesis

Williamson Symplectic-Eigenvalue Threat Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1540_tuesday_local_williamson_symplectic_threat.md`.

Working thesis: Pairs squares with rule-derived next-action 'momenta', builds a 2n x 2n SPD operator on the resulting phase space, and reads off symplectic eigenvalues from Williamson normal form M = S^T D S (S in Sp(2n, R)); a phase-space invariant outside O(2n).

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.

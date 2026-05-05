# Math Thesis

Sylvester Tactical Coupling Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1500_tuesday_local_sylvester_tactical_coupling.md`.

Working thesis: Couples a learned attacker operator A and defender operator B through the Sylvester equation A X + X B = C; classifies puzzles from properties of the unique solution X (singular spectrum, resonance gap min |lambda_i(A) + mu_j(B)|).

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.

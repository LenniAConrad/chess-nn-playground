# Math Thesis

Schur-Complement Defender Elimination Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1505_tuesday_local_schur_complement_defender.md`.

Working thesis: Block-partitions a learned PSD interaction matrix into attacker/defender squares and classifies from the Schur complement S = D - B^T A^{-1} B; Haynsworth inertia of S exposes residual defensive insolvency.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.

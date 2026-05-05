# Math Thesis

Riccati Optimal-Defense Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1600_tuesday_local_riccati_optimal_defense.md`.

Working thesis: Treats each board as an LQR control problem; solves the algebraic Riccati equation A^T P + P A - P B R^{-1} B^T P + Q = 0 via Schur of the Hamiltonian H = [[A, -BR^{-1}B^T],[-Q,-A^T]]; optimal-defense cost J* = trace(P) and closed-loop spectral margin separate puzzles from non-puzzles.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.

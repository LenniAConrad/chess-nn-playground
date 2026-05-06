# Math Thesis

Riccati Optimal-Defense Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1600_tuesday_local_riccati_optimal_defense.md`.

Working thesis: Treats each board as an LQR control problem; solves the algebraic Riccati equation A^T P + P A - P B R^{-1} B^T P + Q = 0 via Schur of the Hamiltonian H = [[A, -BR^{-1}B^T],[-Q,-A^T]]; optimal-defense cost J* = trace(P) and closed-loop spectral margin separate puzzles from non-puzzles.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

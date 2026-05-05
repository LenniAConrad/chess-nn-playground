# Math Thesis

Lyapunov Stability Threat Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1520_tuesday_local_lyapunov_threat_stability.md`.

Working thesis: Treats each board as autonomous dynamics dot x = A x and solves A^T P + P A = -Q for a chess-derived weighting Q; uses inertia, condition number, and trace of P as a quadratic stability certificate distinct from the controllability Gramian.

This registered implementation routes the thesis through the `linear_algebra`
mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving
all source/CRTK metadata for reporting. See the source packet for the full
mathematical derivation, ablations, and falsification criteria.

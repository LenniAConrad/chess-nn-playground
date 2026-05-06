# Math Thesis

Lyapunov Stability Threat Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1520_tuesday_local_lyapunov_threat_stability.md`.

Working thesis: Treats each board as autonomous dynamics dot x = A x and solves A^T P + P A = -Q for a chess-derived weighting Q; uses inertia, condition number, and trace of P as a quadratic stability certificate distinct from the controllability Gramian.

Scaffold-only implementation notice: This folder records the thesis and a shared `ResearchPacketProbe` scaffold only. It is not a completed bespoke implementation of the markdown architecture and must remain `implementation_kind: shared_probe_variant` until matching model code replaces the shared probe.

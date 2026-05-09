# Math Thesis

Riccati Optimal-Defense Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1600_tuesday_local_riccati_optimal_defense.md`.

Working thesis: Treats each board as an LQR control problem; solves the
continuous algebraic Riccati equation `A^T P + P A - P B R^{-1} B^T P +
Q = 0` via the Hamiltonian `H = [[A, -B R^{-1} B^T], [-Q, -A^T]]`. The
unique stabilizing PSD solution `P` is recovered from the stable
invariant subspace of `H` (the `r` eigenvectors with smallest real
eigenvalue) as `P = V_2 V_1^{-1}`. The optimal LQR cost
`J* = trace(P)`, the optimal feedback gain `K = R^{-1} B^T P`, the
closed-loop spectral margin `min Re spec(A - B K)`, and the
Hamiltonian's near-imaginary mode count provide the puzzle-classification
signal.

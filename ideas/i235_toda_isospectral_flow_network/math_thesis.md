# Math Thesis

Toda Isospectral Flow Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1620_tuesday_local_toda_isospectral_flow.md`.

Working thesis: Build a learned symmetric tridiagonal chess operator
`L_0 = tridiag(b, a, b)` and integrate the Toda Lax flow

    dot L = [L, B(L)],   B(L) = L_- - L_+,

with `L_-` / `L_+` the strictly lower / upper triangular parts. By Symes'
theorem the flow is isospectral, so the eigenvalues of `L` are conserved and
the per-iteration trace power sums `Tr(L^k)` (Manakov integrals) are
constants of motion. The diagonal sorts exponentially toward the descending
spectrum of `L_0`; off-diagonal entries decay at rates governed by adjacent
eigenvalue gaps.

The classifier reads off three families of features that distinguish puzzle-
likeness without ever calling an eigendecomposition:

  * sorting rate / final sortedness of the diagonal,
  * Manakov drift `Tr(L_T^k) - Tr(L_0^k)` (a numerical fidelity diagnostic
    that should remain near zero),
  * slowest-decaying off-diagonal residue and the resulting smallest-gap
    proxy `gap ~= -min_i log(b_i(T) / b_i(0)) / T`.

The bespoke implementation of this thesis lives in
`src/chess_nn_playground/models/toda_isospectral_flow.py` and is registered as
`toda_isospectral_flow_network`; the idea folder's `model.py` is a thin
wrapper around `build_toda_isospectral_flow_network_from_config`.

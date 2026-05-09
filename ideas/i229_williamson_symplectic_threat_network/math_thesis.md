# Math Thesis

Williamson Symplectic-Eigenvalue Threat Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1540_tuesday_local_williamson_symplectic_threat.md`.

Working thesis: Pairs squares with rule-derived next-action 'momenta',
builds a `2n x 2n` SPD operator `M` on the resulting phase space, and
reads off symplectic eigenvalues from Williamson normal form
`M = S^T D S` (`S in Sp(2n, R)`, `D = diag(d_1, ..., d_n, d_1, ...,
d_n)`); a phase-space invariant outside `O(2n)` because `Sp(2n, R)`
preserves the symplectic form `J`, not the Euclidean one.

Stable algorithm: `M^{1/2}` via `eigh`, `K = M^{1/2} J M^{1/2}` (skew),
eigenvalues of `K^T K` are `{d_i^2}` each with numerical multiplicity 2;
sort, pair-average, and take square roots to recover the symplectic
spectrum. The forward pass also returns the ordinary spectrum of `M` for
the falsifier `ordinary_eigvals_swap` and `symplectic_entropy`,
`heisenberg_slack`, and adjacent gaps over the symplectic spectrum.

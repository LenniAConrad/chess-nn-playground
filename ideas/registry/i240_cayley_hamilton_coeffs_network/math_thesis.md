# Math Thesis

Cayley-Hamilton Coefficient Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-05-05_1720_tuesday_local_cayley_hamilton_coeffs.md`.

Working thesis: Extracts characteristic polynomial coefficients (c_1, ..., c_r) of a learned r x r chess operator via Faddeev-LeVerrier recursion (no eigendecomposition). c_k = (-1)^k e_k(spec) are signed elementary symmetric polynomials of eigenvalues -- combinatorially distinct from raw eigvals and gradient-stable near degenerate spectra.

This idea is **implemented as a bespoke torch module** at
`src/chess_nn_playground/models/trunk/cayley_hamilton_coeffs.py`
(class `CayleyHamiltonCoefficientNetwork`, builder `build_cayley_hamilton_coeffs_from_config`); not routed
through the generic ResearchPacketProbe.

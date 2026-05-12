# Architecture

`Toda Isospectral Flow Network` is a bespoke implementation of the Toda lattice
Lax flow over a learned chess operator.

- Mechanism family: `linear_algebra`.
- Input: board tensor only (`simple_18`); CRTK/source metadata is reporting-only.
- Board trunk: `BoardConvStem` of configurable width and depth, pooled with mean
  and max along the spatial axes.
- Operator construction: a linear projection turns the pooled features into
  `2n - 1` parameters (n diagonals plus n - 1 positive off-diagonals), forming a
  symmetric tridiagonal `L_0 = tridiag(b, a, b)`.
- Lax flow: explicit Euler integration of `dot L = [L, B(L)]` with
  `B(L) = L_- - L_+` (strictly lower minus strictly upper triangular). The
  iterates are re-symmetrised at every step to absorb roundoff drift.
- Diagnostics:
  * sorting score of the diagonal at the final time;
  * Manakov drift `Tr(L_T^k) - Tr(L_0^k)` for `k = 2..K` (zero in the
    continuous limit; deviation reports numerical fidelity);
  * off-diagonal magnitudes and per-step decay rates;
  * smallest spectral gap estimate `gap = -min_i log(b_i(T) / b_i(0)) / T`.
- Head: pooled features concatenated with the diagnostics feed a LayerNorm +
  GELU MLP that returns one puzzle logit and a dictionary of diagnostic
  tensors.

The flow integrates the standard non-periodic Toda lattice. As `T -> inf` the
diagonal sorts to the descending order of `eig(L_0)`; the off-diagonals decay at
rates set by adjacent eigenvalue gaps. This makes the diagnostics correlated
with the spectral structure of the learned operator without ever calling an
eigendecomposition - the head only sees the lattice variables and the conserved
power sums.

## Implementation Binding

- Registered model name: `toda_isospectral_flow_network`
- Source implementation file: `src/chess_nn_playground/models/toda_isospectral_flow.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i235_toda_isospectral_flow_network/model.py`

The wrapper builds the bespoke
`chess_nn_playground.models.toda_isospectral_flow.TodaIsospectralFlowNetwork`
through `build_toda_isospectral_flow_network_from_config(config["model"])`; it
no longer delegates to `ResearchPacketProbe` or
`build_research_packet_probe_from_config`.

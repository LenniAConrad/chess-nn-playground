# Math Thesis

Pivot Trace Elimination Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.

Batch candidate rank: `6`.

## Thesis

Gaussian elimination exposes interaction structure through pivot sizes,
residual norms, and Schur updates. A chess position is encoded into
`K = 12` learned group summaries (piece-type groups, side roles, line
groups, king-region groups, center / edge), assembled into a small
symmetric interaction matrix

```
M_ij = (W_L g_i)^T (W_R g_j),   M = (M + M^T) / 2 + lambda I,
```

and pushed through a *fixed-order* differentiable Gaussian elimination

```
pivot_t       = softplus(M_tt) + eps
row_update_t  = M_{t+1:, t} / pivot_t
M_{t+1:, t+1:} -= row_update_t outer M_{t, t+1:}.
```

The log-pivot sequence, off-diagonal update norms, residual decay curve,
final residual norm, and condition-like ratio (`log(running_max_pivot /
running_min_pivot)`) form the compact algebraic signature read out by
the classifier. Following the packet's implementation note we do not use
learned pivoting; the elimination order is the canonical group order so
that the `random_elimination_order` ablation cleanly tests whether
semantic order matters.

## Why It Is Distinct

- Not Matrix-Pencil: no generalised eigenvalues are used as the main
  readout (provided as the `matrix_pencil_control` ablation only).
- Not Polar-Procrustes: no orthogonal alignment.
- Not Schur-Ray: no line-incidence Schur solve over rays.
- Not determinantal volume: the determinant is a single scalar in the
  trace, not the readout (provided as the `determinant_only` ablation).

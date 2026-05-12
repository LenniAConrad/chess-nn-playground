# Math Thesis

Schur-Ray Line Algebra Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2127_friday_shanghai_schur_ray_line_algebra.md`.

Working thesis: a chess board is naturally described by 64 squares plus a
fixed family of 46 rays (8 ranks, 8 files, 15 diagonals, 15 anti-diagonals).
A puzzle classifier built around this structure should recover the
equilibrium square field of the linear system

```
(D(x) + U(x) C U(x)^T) z = D(x) b(x)
```

where `D(x) = diag(d(x))` is a positive per-square data weight, `b(x)` is
the per-square target, `U(x) in R^{64 x R}` is a board-conditioned ray
incidence (the 46 rays expanded into a low-rank `R = H r` mode basis),
and `C = diag(c) > 0` is a positive diagonal line coupling. By the
Sherman-Morrison-Woodbury identity, the equilibrium square field is

```
z = b - D^{-1} U (U^T D^{-1} U + C^{-1})^{-1} U^T b,
```

so the 64x64 square solve reduces to an `R x R` Schur-complement Cholesky
solve in line space. The thesis is that the equilibrium correction
`z - b`, line-coefficient energy, and Schur log-det/trace carry
puzzle-vs-non-puzzle signal that generic CNN pooling does not extract.

The implementation realizes this exactly: a CoordinateBoardStem produces
square features; per-head data fields `(b, d, g)` parameterize `b` and
`D` and a blocker gate `g`; a BoardConditionedLineModes module produces
ray modes `M(x)` from a fixed rank/file/diagonal incidence; the gated
square modes form `U(x)`; and a Cholesky factorization of
`U^T D^{-1} U + C^{-1} + jitter I` solves the Schur system. The classifier
sees the Schur diagnostics together with CNN and material summaries.

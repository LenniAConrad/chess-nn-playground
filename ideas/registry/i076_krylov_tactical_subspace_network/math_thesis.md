# Math Thesis

Krylov Tactical Subspace Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_2000_saturday_shanghai_krylov_tactical_subspace.md`.

Working thesis: chess tactics are repeated propagation of pressure
through legal geometry, and a tactical position can be characterised
by the Krylov subspaces

```
K_m(A, v_r) = span{v_r, A v_r, A^2 v_r, ..., A^{m-1} v_r}
```

generated when a learned but chess-structured linear operator `A(X)`
is applied repeatedly to role-conditioned seed vectors `v_r` for
roles `attack`, `defense`, `king_zone`, `high_value_target`,
`blocker`, and `tempo`.

The bespoke model exposes the math of this thesis:

- The operator is `A = sum_g gate_g(X) * mask_g + U(X) V(X)^T` with
  fixed deterministic geometry masks (ray, knight, pawn, king, defense)
  and a low-rank board-conditioned context update.
- A differentiable modified Gram-Schmidt Arnoldi block produces the
  orthonormal basis `Q_r` and the upper-Hessenberg projection `H_r`
  per role.
- The puzzle logit is read from Ritz singular values of `H_r`,
  Arnoldi residual norms, growth curves `||A^k v_r||`, basis energy
  near the side-to-move king and opposing high-value pieces, and the
  principal angles between role subspaces (singular values of
  `Q_a^T Q_b`).

The core hypothesis: true puzzles produce distinctive operator
dynamics — rapid attacker-Krylov concentration on king and
high-value-target squares, weak defender-Krylov coverage of those
directions, and high attacker/defender subspace conflict. Near
puzzles share first-order pressure but diffuse, cancel, or align with
defender coverage at higher Krylov order.

The classifier does not consume engine, source, verification, or
CRTK metadata; the only inputs are deterministic chess geometry and
the current `simple_18` board tensor.

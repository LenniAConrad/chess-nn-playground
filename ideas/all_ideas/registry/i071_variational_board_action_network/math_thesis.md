# Math Thesis

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2146_friday_shanghai_variational_board_action.md`.

The thesis is that puzzle-like chess positions often look locally out of equilibrium
under an ordinary board field action. The model learns a field \(u\) over the current
8x8 board and an action density

```text
A[u; x] = sum_s L_theta(u(s), grad u(s), x(s)).
```

For a stationary field, the discrete Euler-Lagrange residual should be small:

```text
R = dL/du - div(dL/d grad u).
```

Tactical positions can create localized variational defects: king-zone pressure,
overloaded defenders, pinned pieces, or sharp force discontinuities. This model makes
that defect explicit by computing residual maps from learned fields and fixed board
finite differences, then classifying puzzle-likeness from the residual summaries and
localized residual maps.

The implementation uses the packet's recommended first version: a force-head
approximation to `dL/du` instead of exact autograd through a learned potential scalar.
The finite-difference geometry, positive stiffness maps, divergence adjoints,
action-energy terms, and residual-map bottleneck are explicit model features.

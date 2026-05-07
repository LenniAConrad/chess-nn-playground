# Math Thesis

Legal Automorphism Quotient Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0731_tuesday_los_angeles_orbit_quotient.md`.

Working thesis: A chess puzzle-likeness classifier should quotient out
the exact current-board automorphisms of chess rules - file mirror and
color/side flip - so that the supervised head cannot spend capacity on
orientation artifacts that do not change whether a position is
puzzle-like.

## Group action

Let ``b in R^{18 x 8 x 8}`` be a simple_18 tensor. Define two commuting
involutions on the simple_18 channel layout:

- ``m`` (**file mirror**): flip files ``a <-> h``, swap king-side and
  queen-side castling rights for each color, mirror the en-passant
  file, keep ranks, piece colors, and side-to-move.
- ``q`` (**color/rank flip**): reflect ranks ``1 <-> 8``, swap White
  and Black piece planes, toggle the side-to-move bit, swap
  White<->Black castling rights preserving king/queen side, and reflect
  the en-passant rank.

These commute and generate

```
G = <m, q> ~ C2 x C2 = {e, m, q, mq}.
```

No 90-degree rotation or naked vertical flip without color swap is
assumed because pawns, castling, en-passant, and side-to-move would
break those operations.

## Reynolds invariant latent

Let ``phi_theta : R^{C x 8 x 8} -> R^{d}`` be a shared CNN encoder.
The Reynolds projector

```
P_G phi_theta(s) = (1 / |G|) sum_{g in G} phi_theta(kappa_E(g . s))
```

is exactly invariant under ``G``. Decomposing over the four C2 x C2
characters yields

```
P_chi phi_theta(s) = (1 / |G|) sum_{g in G} chi(g) phi_theta(kappa_E(g . s)).
```

The classifier consumes only the trivial-character component
``z_0(s) = P_{chi_0} phi_theta(s)``. The three nontrivial-character
components are surfaced as diagnostics. The optional regularizer

```
R_char(s) = sum_{chi != chi_0} || P_chi phi_theta(s) ||_2^2
```

penalizes encoder responses that fail to align across the orbit; the
`character_penalty` diagnostic returned by the model is exactly this
quantity, ready to be scaled by `char_penalty_weight` in the trainer.

## Why it matters

By Jensen's inequality on the convex cross-entropy loss, on the
symmetrized empirical distribution the orbit-averaged Reynolds
predictor cannot be worse than the per-orbit average loss of the
unaveraged predictor (assuming label invariance under ``G``). The
forward pass is exact in the sense that the trivial-character latent is
provably invariant under all four legal automorphisms, while the
nontrivial-character norms vanish exactly when the encoder agrees
across the orbit.

## What remains hypothesized

- That puzzle-likeness labels in this dataset are close enough to
  invariant under ``G`` for quotienting to help.
- That important shortcut correlations are non-invariant under ``G``.
- That improved invariance especially helps verified near-puzzles.
- That the chess-rule semantics of ``G`` matter - this is falsified by
  the central randomized-orbit ablation (replace ``G`` with four random
  square permutations preserving channel counts).

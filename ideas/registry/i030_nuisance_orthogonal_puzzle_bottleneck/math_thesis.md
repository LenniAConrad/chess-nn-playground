# Math Thesis

Nuisance-Orthogonal Puzzle Bottleneck (NOPB).

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0508_tuesday_local_nuisance_orthogonal.md`.

## Working Thesis

A chess puzzle-like position should remain recognizable after the model's
learned latent representation is explicitly projected away from deterministic
material/phase/king/castling/side-to-move nuisance directions, because tactical
interest is partly a *residual* structural property rather than a
material-profile shortcut.

## Setup

Let ``B`` denote a legal board encoded as the ``simple_18`` tensor
``X(B) in R^{18 x 8 x 8}``. Let ``Y in {0, 1}`` be the puzzle-binary label
(fine label ``2`` maps to ``Y=1``; fine labels ``0`` and ``1`` map to
``Y=0``).

For a mini-batch of size ``b`` we materialise:

- ``H = f_theta(X) in R^{b x d}``: the learned latent produced by a compact
  convolutional residual trunk;
- ``n(B) in R^{m}``: a deterministic nuisance vector (material counts, phase,
  side-to-move, castling, en-passant, king coordinates, pawn-file profile,
  coarse occupancy marginals);
- ``Q = rho(n(B)) in R^{b x k}``: a fixed normalized nuisance feature matrix
  built from ``n`` via a deterministic random projection of pairwise products
  followed by a non-affine LayerNorm.

## Operator

We classify
``logits = c_phi(Z)`` where

```
Z = H_c - gamma * Q_c (Q_c^T Q_c + lambda I_k)^{-1} Q_c^T H_c
```

with ``H_c`` and ``Q_c`` mini-batch-centred. Defaults: ``gamma = 1``,
``lambda = 1e-3``.

## Proposition (Empirical Nuisance Orthogonality)

Assume ``lambda = 0``, ``rank(Q_c) = k`` and ``gamma = 1``. Then
``Q_c^T Z = 0``, i.e. every coordinate of ``Z`` has zero empirical linear
covariance with every nuisance feature on that mini-batch.

**Proof.** Substituting,
``Q_c^T Z = Q_c^T H_c - Q_c^T Q_c (Q_c^T Q_c)^{-1} Q_c^T H_c = 0``.
For ``lambda > 0`` the residual obeys
``Q_c^T Z = lambda (Q_c^T Q_c + lambda I)^{-1} Q_c^T H_c``,
which is small whenever ``lambda`` is much smaller than the eigenvalues of
``Q_c^T Q_c``.

## Variational Characterisation

The projection is the closed-form solution to
``Z* = argmin_Z ||H - Z||_F^2`` subject to ``Q^T Z = 0``; ``lambda > 0`` is the
ridge-stabilised version of the same residualisation.

## Falsification

The smallest central falsification is ``gamma = 0`` with the trunk, head,
optimizer, batch, and deterministic nuisance extraction unchanged: this leaves
the same parameter budget but removes the residualisation operator. Shuffled
``Q``, columnwise-permuted ``Q``, random rank-matched ``Q``, and a
nuisance-only classifier provide the auxiliary semantics-destroying ablations.

## What Remains Hypothesised

It is not proved that puzzle-likeness is independent of material/phase/side-to-
move on this dataset, only that the projection removes empirical linear
dependence between ``Z`` and a fixed deterministic ``Q`` on each mini-batch.
Whether this constraint actually improves binary discrimination, near-puzzle
recall at matched FPR, or generalisation across split tags is the empirical
question the trainer answers.

## Implementation Note

The bespoke implementation lives at
`src/chess_nn_playground/models/nuisance_orthogonal_puzzle_bottleneck.py` and
is wired into the project model registry under the name
`nuisance_orthogonal_puzzle_bottleneck`. The idea-local
`ideas/registry/i030_nuisance_orthogonal_puzzle_bottleneck/model.py` is a thin builder
wrapper around that implementation.

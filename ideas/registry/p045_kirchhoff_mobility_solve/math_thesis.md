# Math Thesis

Source: `ideas/research/primitives/external_40_symmetric_coalition_resolvent_primitives.md`
(Section 4 `primitive_kirchhoff_mobility_solve`; promoted from #4
because the file's other proposals overlap with this batch's already-
covered families).

## Working thesis

For a position with simple_18 board tensor:

1. Build the 8x8 grid vertex-edge incidence ``D in R^{64 x 112}``
   (oriented horizontal + vertical nearest-neighbour edges).
2. Pool the i193 spatial features ``S in R^{B x 2C x 8 x 8}`` to a
   per-square representation ``X in R^{B x 64 x 2C}``.
3. Per-edge endpoint feature: gather ``X`` at head and tail squares,
   concatenate.

       e_kl in R^{B x 4C}.
4. Conductance head:

       c_{kl} = softplus(MLP_2(e_kl)).
5. Per-square source/sink:

       s_v = Linear_source(X_v).
6. Laplacian assembly:

       L_b = D^T diag(c_b) D + lambda I.

   Both terms are SPD; the shift ``lambda`` ensures strict positive-
   definiteness and a stable solve. (Note that ``D^T diag(c) D`` has
   a one-dimensional kernel spanned by the constant vector; the shift
   removes that singularity.)
7. Solve the SPD system per batch:

       u_b = L_b^{-1} s_b in R^{64 x source_channels}.

   This is the exact electrical-potential equilibrium of the
   conductance system, NOT a fixed-iteration approximation.
8. Output projection ``Y_v = u_v W_o``, pooled with mean + max per
   channel, concatenated with the i193 trunk joint pool, projected to
   `primitive_delta_raw`. Gate is a sigmoid MLP over joint +
   ``kms_potential_norm`` + ``kms_conductance_mean``.
9. Output: ``final_logit = base_logit + primitive_gate *
   primitive_delta_raw``.

## Why this matters

King safety and fortress evaluation are inherently long-range and
bottleneck-aware: one occupied square can flip the connectivity of a
region. A conv stack with k layers can propagate information at most
k squares; the resolvent integrates the entire grid in one solve.
The conductance ``c_kl = 0`` (or very small) breaks the edge, and the
potential ``u`` reroutes around the obstruction. This is the right
inductive bias for "blocked diagonal", "escape-square cage", and
"isolated pawn island" patterns.

## What is actually proven

- ``D^T diag(c) D`` is positive semi-definite for c >= 0; the
  ``lambda I`` shift makes the Laplacian SPD so
  ``torch.linalg.solve`` always succeeds.
- The solve is differentiable through PyTorch's implicit reverse
  mode (``solve`` already implements the adjoint).
- ``uniform_conductance`` collapses to a fixed linear projection of
  the source; ``diagonal_only`` zeroes the Laplacian term entirely,
  giving ``u = s / lambda`` -- both are clean falsifiers.

## What is only hypothesized

That the input-dependent conductance carries chess-specific bottleneck
information not already encoded by the i193 trunk.

## Failure cases

1. *Hidden rebrand of conv*: tested by `uniform_conductance` (the
   resolvent of a fixed Laplacian is a fixed linear map of ``s``).
2. *Source is enough*: tested by `diagonal_only`, which drops the
   Laplacian term entirely; the head sees ``u = s / lambda`` which is
   just a per-square reweighting of the source.
3. *Ill-conditioning*: ``shift = 1e-2`` keeps the smallest eigenvalue
   bounded above zero.
4. *Throughput regression*: O(B * 64^3) solves are well within budget
   on RTX 3070-class hardware. If batch size > 512 becomes pressure,
   precomputed Cholesky factorisation is the next upgrade.

## Falsifier

- `uniform_conductance` — primary. Replace the learned positive
  conductance with all-ones. Tests whether the input-dependent metric
  is load-bearing.
- `diagonal_only` — secondary. Drop the Laplacian term entirely; the
  head sees ``u = s / lambda``. Tests whether the Laplacian-resolvent
  structure matters beyond a per-square source readout.
- `shuffle_conductance` — in-batch permutation of the conductance
  vector. Decouples conductance from board geometry.
- `zero_source` — zero the source term; ``u = 0`` everywhere. Sanity
  check that the head responds to the source signal.

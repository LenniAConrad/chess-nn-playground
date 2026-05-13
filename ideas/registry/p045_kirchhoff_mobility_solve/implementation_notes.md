# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/kirchhoff_mobility_solve.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Idea-local wrapper: `ideas/registry/p045_kirchhoff_mobility_solve/model.py`.
- Registry key: `kirchhoff_mobility_solve`.
- Source primitive: `ideas/research/primitives/external_40_symmetric_coalition_resolvent_primitives.md`
  (Section 4 ``primitive_kirchhoff_mobility_solve``).

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
The 8x8 grid vertex-edge incidence is precomputed once and stored as
a non-persistent buffer ``self.d_matrix``.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The SPD solve is differentiable via PyTorch's ``torch.linalg.solve``.
Source and conductance heads carry gradients; trunk diagnostics fed
to the gate are detached.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_applied`` / ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution``
- ``kms_potential_norm`` -- RMS of the per-vertex potential
- ``kms_conductance_mean`` -- mean conductance per sample
- ``kms_source_norm`` -- RMS of the source term
- ``trunk_<name>`` for every diagnostic the i193 trunk produced
- ``mechanism_energy`` augmented with ``kms_potential_norm.detach()``
- ``proposal_profile_strength`` = ``|delta| * gate_entropy``

## Ablation modes

See ``ALLOWED_ABLATIONS``. Primary falsifier is ``uniform_conductance``
(set c = 1 so the Laplacian becomes a fixed graph Laplacian).
``diagonal_only`` drops the Laplacian term and is the secondary
falsifier. ``shuffle_conductance`` decouples conductance from board
geometry. ``zero_source`` is a sanity check.

## Numerical notes

- Conductance is ``softplus(.) >= 0`` with a ``clamp_min(1e-3)`` floor
  inside ``kirchhoff_resolve`` to prevent ill-conditioning when the
  network initialises with strongly negative conductance logits.
- The ``+ shift * I`` shift (default 1e-2) makes the Laplacian SPD;
  without it, ``D^T diag(c) D`` has a one-dimensional kernel along
  the constant vector.
- ``torch.linalg.solve`` handles batched SPD systems; backward
  propagates through implicit differentiation. No explicit adjoint
  bookkeeping is needed.

## Production upgrade path

- Sparse Cholesky factorisation for ``L_b`` to reduce O(N^3) cost.
- Sherman-Morrison rank-2 update for bounded-change inference.
- Conjugate-gradient with preconditioning for larger graphs.

Deferred until keep-decision is in.

# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/weighted_hodge_projector.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Idea-local wrapper: `ideas/registry/p044_weighted_hodge_projector/model.py`.
- Registry key: `weighted_hodge_projector`.
- Source primitive: `ideas/research/primitives/external_39_orbit_irrep_hodge_projection_primitives.md`
  (Section 2 ``primitive_weighted_hodge_projector``).

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
The 8x8 grid complex (vertices = 64 squares, edges = 112 oriented
nearest-neighbour pairs, faces = 49 unit squares) is precomputed once
in `_build_incidence_matrices` and stored as non-persistent buffers
``self.d0`` and ``self.d1``.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The two SPD solves are differentiable via PyTorch's
`torch.linalg.solve` (which implements implicit reverse mode). The
flow and metric heads carry gradients; the trunk diagnostics fed to
the gate are detached.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_applied`` / ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution``
- ``whp_gradient_energy`` / ``whp_curl_energy`` / ``whp_harmonic_energy``
- ``whp_flow_energy`` -- RMS of the raw flow
- ``whp_weight_mean`` -- mean of the softplus metric per sample
- ``trunk_<name>`` for every diagnostic the i193 trunk produced
- ``mechanism_energy`` augmented with the sum of component energies
- ``proposal_profile_strength`` = ``|delta| * gate_entropy``

## Ablation modes

See ``ALLOWED_ABLATIONS``. Primary falsifier is ``uniform_metric``
(force ``W = I`` so the projection becomes a fixed linear map).
Per-component ablations (``drop_curl`` / ``drop_gradient`` /
``drop_harmonic``) confirm that each component carries some signal.
``shuffle_edge_flow`` decouples flow values from edge geometry.

## Numerical notes

- Edge metric ``w = softplus(metric_head(.))`` is positive; we clamp it
  to >= 1e-3 before the solves to prevent ill-conditioning when the
  network initialises with strongly negative metric logits.
- Both Laplacian solves use ``+ solve_eps * I`` (default 1e-2) for
  Cholesky / LU stability. The eps shift is the only non-orthogonal
  perturbation in the decomposition; the residual ``H = R - Cr`` is
  exactly what is left over.
- ``D_1^T D_0^T == 0`` is built into the incidence orientations
  (top -> right -> -bottom -> -left).

## Production upgrade path

- Replace the dense solves with sparse / Cholesky factorisations when
  ``flow_channels`` grows.
- Cache the LU factorisations across small bounded edits — relevant
  only at engine inference time, not for the scout training loop.
- For an exact orthogonal decomposition, replace the eps-shifted solves
  with pseudo-inverses; deferred until the keep-decision is in.

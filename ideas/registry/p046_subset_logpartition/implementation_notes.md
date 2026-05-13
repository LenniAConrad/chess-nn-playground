# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/subset_logpartition.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Idea-local wrapper: `ideas/registry/p046_subset_logpartition/model.py`.
- Registry key: `subset_logpartition`.
- Source primitive: `ideas/research/primitives/external_41_orbit_stabilizer_subset_logpartition_primitives.md`
  (Section 2 ``primitive_subset_logpartition``).

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
Per-square log-weights are computed inline as
``LayerNorm(Linear([piece_planes; stm]))``. Occupancy mask is the
clamped sum of the 12 piece-presence planes.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The log-semiring scan is fully differentiable via autograd through
``torch.logaddexp``. Trunk diagnostics fed to the gate are detached.
The joint pool feature used in the delta head is not detached so the
delta can co-train the trunk pool path.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_applied`` / ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution``
- ``slpt_active_mean`` -- mean active-token mask per sample
- ``slpt_logpartition_norm`` -- RMS of the (clipped) log-partition tensor
- ``slpt_y1`` / ``slpt_y2`` / ``slpt_y3`` -- per-degree channel mean log-partition values
- ``trunk_<name>`` for every diagnostic the i193 trunk produced
- ``mechanism_energy`` augmented with ``slpt_logpartition_norm.detach()``
- ``proposal_profile_strength`` = ``|delta| * gate_entropy``

## Ablation modes

See ``ALLOWED_ABLATIONS``. Primary falsifier is ``k1_only`` (collapses
K to 1 -> logsumexp pool). Rule-feature falsifiers are
``uniform_mask`` (mask all ones) and ``shuffle_mask`` (batch-permuted
mask). ``shuffle_tokens`` verifies token-order invariance.

## Numerical notes

- Inactive tokens are set to ``log(0) = -1e9`` (NEG_INF) so they
  cannot contribute to any subset under the recurrence.
- Log-weights are real-valued (no tanh activation) -- the operator
  expects log-domain inputs. ``LayerNorm`` bounds their per-channel
  scale.
- The final log-partition is clipped to ``[-30, 30]`` before the head
  input to bound the dynamic range across positions with very
  different active-token counts. The clip is applied only at the head
  input; the scan itself runs in unrestricted log-domain.
- Autograd handles the ``logaddexp`` backward; no custom subset-
  marginal implementation is needed.

## Production upgrade path

- Fused CUDA / Triton log-semiring scan for inference-time speed.
- O(K) bounded-edit update using polynomial factor removal.

Deferred until keep-decision is in.

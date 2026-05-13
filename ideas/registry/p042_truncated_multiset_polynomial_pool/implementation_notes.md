# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/truncated_multiset_polynomial_pool.py`.
- Shared trunk helper: `src/chess_nn_playground/models/primitives/trunk_features.py`
  (`trunk_joint_features`).
- Idea-local wrapper: `ideas/registry/p042_truncated_multiset_polynomial_pool/model.py`.
- Registry key: `truncated_multiset_polynomial_pool`.
- Source primitive: `ideas/research/primitives/external_37_truncated_multiset_polynomial_rook_matching_primitives.md`
  (Section 1 `primitive_truncated_multiset_polynomial_pool`; first-listed proposal).

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
Per-square latents are computed inline as
``tanh(LayerNorm(Linear([piece_planes; stm])))``.
The occupancy mask is the clamped sum of the 12 piece-presence planes.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The polynomial scan is fully differentiable. Trunk diagnostics fed
into the gate are detached. The joint pool feature used in the delta
head is not detached so the delta can co-train the trunk's pool path.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_applied`` / ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution`` (gate * delta)
- ``tmpp_active_mean`` — mean active-token mask per sample
- ``tmpp_coeff_norm`` — RMS of the stacked coefficient tensor
- ``tmpp_coeff_e1`` / ``tmpp_coeff_e2`` / ``tmpp_coeff_e3`` — per-degree
  channel-RMS values (only the degrees up to ``degree`` are emitted)
- ``trunk_<name>`` for every diagnostic the i193 trunk produced
- ``mechanism_energy`` augmented with ``tmpp_coeff_norm.detach()``
- ``proposal_profile_strength`` = ``|delta| * gate_entropy``

## Ablation modes

See ``ALLOWED_ABLATIONS``. Primary falsifier is ``first_order_only``
(collapses K to 1 ⇒ DeepSets-style weighted sum). Rule-feature
falsifiers are ``uniform_mask`` (mask all ones) and ``shuffle_mask``
(batch-permuted mask). ``shuffle_tokens`` verifies the scan's order
invariance.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` with a fused polynomial scan that
no existing builder reproduces. It does not call
``build_research_packet_probe_from_config`` and does not delegate to a
shared CNN / MLP baseline builder.

## Numerical notes

- ``tanh`` keeps the per-token latent in (-1, 1); the LayerNorm
  upstream centres it before tanh.
- The optional ``coeff_norm`` divisor is an estimate of
  ``binom(active, k)``: it does *not* require a true binomial since
  we only need a magnitude estimate. The factor is computed without
  gradient propagation so it cannot interact pathologically with
  training.
- ``degree`` is capped to 6 in the builder; defaults to 3 per the
  source primitive recommendation. Values above 4 are unsafe without
  log-domain coefficients.

## Production upgrade path

The Python loop over 64 squares is intentional — torch.jit / triton
fusion is the next upgrade, deferred until the keep-decision is in.
For training-time batching there is no observed bottleneck because the
64 outer-loop iterations each fire one Hadamard add over `(B, K, C)`
tensors.

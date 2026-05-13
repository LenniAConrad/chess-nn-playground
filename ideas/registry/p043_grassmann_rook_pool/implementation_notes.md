# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/grassmann_rook_pool.py`.
- Shared helpers:
  - `BoardTokenAttention` from `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
  - `trunk_joint_features` from `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Idea-local wrapper: `ideas/registry/p043_grassmann_rook_pool/model.py`.
- Registry key: `grassmann_rook_pool`.
- Source primitive: `ideas/research/primitives/external_38_polynomial_ledger_grassmann_rook_primitives.md`
  (Section `primitive_grassmann_rook_pool`; second-ranked).

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
Attacker / defender tokens are pooled from the i193 spatial feature
map; per-token validity is a learned sigmoid mask.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The matching-coefficient scan is fully differentiable. Trunk
diagnostics fed into the gate are detached. The joint pool feature
used in the delta head is not detached so the delta can co-train the
trunk pool path.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_applied`` / ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution`` (gate * delta)
- ``grmp_attacker_count`` / ``grmp_defender_count`` — sum of validity masks per sample
- ``grmp_coeff_norm`` — RMS of the stacked coefficient tensor
- ``grmp_coeff_e1`` / ``grmp_coeff_e2`` / ``grmp_coeff_e3`` — per-degree
  channel-RMS values (emitted up to the configured ``degree``)
- ``trunk_<name>`` for every diagnostic the i193 trunk produced
- ``mechanism_energy`` augmented with ``grmp_coeff_norm.detach()``
- ``proposal_profile_strength`` = ``|delta| * gate_entropy``

## Ablation modes

See ``ALLOWED_ABLATIONS``. Primary falsifier is ``drop_exclusion``
(collapses to the flat elementary-symmetric pool, eliminating the
row/column constraint that is the spec's defining property). Score-
channel falsifier is ``scalar_score``. Rule-feature falsifiers are
``shuffle_attackers`` / ``shuffle_defenders`` (batch-permuted tokens).

## Numerical notes

- ``score = tanh(bilinear(...))`` bounds edge magnitudes to (-1, 1).
- Coefficient LayerNorm before the delta head bounds the input range.
- The K=2 closed form may produce small negative values when the
  algebraic cancellation overshoots due to floating-point; downstream
  LayerNorm absorbs this. The K=3 anchor iteration is O(R * C) and is
  acceptable at R=C=8.

## Production upgrade path

- Replace ``nn.Bilinear`` with a low-rank factorisation when ``token_dim``
  grows beyond 64.
- Implement the spec's Sherman-Morrison-style ``(1 + zg)^{-1} = 1 - zg``
  delete update for bounded-change inference; not needed for training.
- For K=4 the closed form becomes unwieldy; switch to a permanent /
  rook-polynomial DP if higher-degree matching becomes important.

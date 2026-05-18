# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/learned_relation_confidence.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Idea-local wrapper: `ideas/registry/p047_learned_relation_confidence/model.py`.
- Registry key: `learned_relation_confidence`.
- Source primitive:
  `ideas/research/primitives/external_42_learned_relation_confidence_primitive.md`.

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
Relation masks are computed inline via the frozen
`BoardStateAdapter` + `TacticalIncidenceBuilder` from i018; both
modules are imported from
`chess_nn_playground.models.trunk.oriented_tactical_sheaf` and their
parameters are explicitly frozen with `requires_grad_(False)`.

Per-square tokens are produced by a small 1x1 conv tower local to the
primitive so it can be ported to non-i193 trunks without dragging in
the i018 `SquareTokenEncoder`.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

- The relation builder and board adapter are frozen.
- The trunk diagnostics fed into the gate MLP are detached.
- The joint pool fed into the delta head is *not* detached so the
  delta can co-train the i193 pool path (matches p046's convention).
- `mechanism_energy` is exported as
  ``trunk_out["mechanism_energy"] + global_mean_conf.detach()``.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_applied`` /
  ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution``
- ``lrc_global_mean_confidence`` -- mean per-relation mean confidence
- ``lrc_mask_density`` -- mean active-edge density across relations
- ``lrc_mean_conf_<rel>``, ``lrc_mass_<rel>``, ``lrc_kept_<rel>``,
  ``lrc_entropy_<rel>`` for each of the 12 i018 relations
- ``trunk_<name>`` for every diagnostic the i193 trunk produced
- ``mechanism_energy`` augmented with
  ``lrc_global_mean_confidence.detach()``
- ``proposal_profile_strength`` = ``|delta| * gate_entropy``
- ``proposal_keyword_count`` = ``R * 4`` (summary width)

## Ablation modes

See `ALLOWED_ABLATIONS`. Primary falsifier is `binary_only` (skip
confidence; summary inputs reduce to raw mask statistics). Rule-feature
falsifiers are `scrambled_mask` (in-batch permute relation masks) and
`shuffle_pieces` (in-batch permute the piece descriptor). The
`gate_only` ablation disables per-edge scoring to test whether per-
relation rescaling is enough.

## Numerical notes

- The per-edge logit is divided by `confidence_temperature` (default
  1.0) before the sigmoid so the head can sharpen confidences without
  re-scaling the bias.
- `relation_bias` is initialised at `confidence_bias_init = 2.0`,
  giving sigmoid(2) ~= 0.88 on untrained active edges; the per-
  relation gate starts at 0.5 (sigmoid(0)).
- The summary uses a soft kept-fraction with sigmoid temperature 0.1
  around threshold 0.5 so the metric is differentiable everywhere; the
  same value is reported as a diagnostic.
- The masked entropy summary clamps the sigmoid to
  `[1e-6, 1 - 1e-6]` to avoid `log(0)`.

## Production upgrade path

- Vectorised relation construction. The current implementation reuses
  the i018 `TacticalIncidenceBuilder`, which contains a Python loop in
  the pin-relation path; a vectorised variant would close that gap.
- Sparse downstream consumption. The current operator emits a dense
  `(B, R, 64, 64)` weighted mask. A rowwise differentiable top-`k`
  pruning step would let downstream consumers operate on sparse
  weighted edges; deferred until the dense version's keep-decision is
  in.

Both are deferred behind the keep-decision on the dense pilot.

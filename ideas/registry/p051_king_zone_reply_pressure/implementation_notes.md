# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/king_zone_reply_pressure.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`
  (used by the cumsum pin detector).
- Shared trunk helper: `src/chess_nn_playground/models/primitives/trunk_features.py`
  (`trunk_joint_features`).
- Idea-local wrapper: `ideas/registry/p051_king_zone_reply_pressure/model.py`.
- Registry key: `king_zone_reply_pressure`.
- Source primitive: `ideas/research/primitives/external_46_king_zone_reply_pressure_primitive.md`.

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is
consumed. The primitive uses absolute (white-first, black-second)
piece planes for the attack-mask construction and re-orients per
side. The side-to-move scalar plane 12 is the only non-piece
plane consulted. Castling / en-passant planes 13..17 are not used.

CRTK metadata, source labels, verification flags, engine scores,
and principal variations are **not** consulted.

## Stop-gradient contract

The operator vector is fully differentiable. Trunk diagnostics
forwarded under the `trunk_<name>` prefix retain their original
gradient state; the `mechanism_energy` augmentation uses
`kzrp_operator_l2.detach()` so the trunk's own mechanism-energy
diagnostic stays causally pinned to the trunk pool norm.

The joint pool feature used in the delta head is **not** detached
so the delta can co-train the trunk's pool path.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_applied`` /
  ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution`` (gate * delta)
- ``kzrp_operator_mean`` / ``kzrp_operator_max`` /
  ``kzrp_operator_l2`` -- aggregate operator-vector statistics
- ``kzrp_asym_score`` (us zone_pressure - them zone_pressure)
- ``kzrp_us_*`` / ``kzrp_them_*`` per-side aux scalars:
  ``zone_pressure``, ``fake_defense_loss``, ``live_escapes``,
  ``sealed_escapes``, ``blocked_escapes``, ``king_attack_mass``,
  ``front_attack_mass``, ``reply_proxy``, ``ring_free_defense``,
  ``net_pressure_mean``, ``net_pressure_max``
- ``trunk_<name>`` for every diagnostic the i193 trunk produced
- ``mechanism_energy`` augmented with ``kzrp_operator_l2.detach()``
- ``proposal_profile_strength`` = ``|delta| * gate_entropy``

## Ablation modes

See ``ALLOWED_ABLATIONS``. The primary falsifier is
``no_front_zone``, which drops the forward-rank zone contribution
and tests whether the front-rank extension is load-bearing on top
of the ring-only zone pressure. Rule-feature falsifiers are
``no_pins`` (π = 0 everywhere), ``uniform_zone_weights`` (replace
`(4, 3, 2)` with uniform 1), ``no_escape_decomp`` (collapse the
three escape-class counts into a total), ``uniform_units``
(uniform attack units), and ``no_asymmetry`` (zero them-side
vector). Numerical-recovery ablations are ``zero_delta``,
``trunk_only``, and ``disable_gate``.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` with a closed-form king-zone
weighted-pressure / escape / reply-proxy builder
(`KingZoneReplyPressureBuilder`) that no existing builder
reproduces. It does not call
``build_research_packet_probe_from_config`` and does not delegate to
a shared CNN / MLP baseline builder.

## Numerical notes

- Zone pressure uses `relu(A - λ_def · D_free)` which clamps the
  per-square contribution to non-negative.
- Fake-defense loss is clamped non-negative (D_nom - D_free is
  mathematically non-negative since `D_free` removes pin-discounted
  mass from `D_nom`, but rounding in mixed precision can produce
  tiny negatives).
- `NaN` and `inf` produced by all-empty / bare-kings positions
  are sanitised at the side-vector boundary via
  `torch.nan_to_num(..., 0.0)`.
- The reply proxy uses `log1p` so it stays finite even on
  bare-kings positions.

## Geometry / orientation note

The simple_18 piece planes are *absolute* (planes 0..5 = white,
6..11 = black). The primitive re-orients to mover-perspective
inside the forward pass by computing per-colour `attack_X_nom` /
`attack_X_free` and selecting with the side-to-move scalar. The
front-rank mask switches between `front_mask_w` (white defender)
and `front_mask_b` (black defender) based on the defender's colour,
so the operator is colour-agnostic from the model's view: a
white-to-move king-pressure position and the colour-flipped
black-to-move position generate identical 32-dim outputs (modulo
mirror-asymmetric pawn attacks).

## Production upgrade path

The attack builder is `einsum` + boolean masks; the side vector is
direct tensor reductions. `torch.compile` should fuse the
dependent ops; this is the spec's preferred path. The phase-2
upgrade ("native i018 `TacticalReadout` integration") and phase-3
upgrade ("BT4 plane augmentation / mixer-native variant") are out
of scope for this idea-folder; the source markdown lists both as
next steps after a positive standalone keep-decision.

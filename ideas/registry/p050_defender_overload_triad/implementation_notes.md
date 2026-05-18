# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/defender_overload_triad.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`
  (used by the cumsum pin detector).
- Shared trunk helper: `src/chess_nn_playground/models/primitives/trunk_features.py`
  (`trunk_joint_features`).
- Idea-local wrapper: `ideas/registry/p050_defender_overload_triad/model.py`.
- Registry key: `defender_overload_triad`.
- Source primitive: `ideas/research/primitives/external_45_defender_overload_triad_primitive.md`.

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is
consumed. The primitive uses absolute (white-first, black-second)
piece planes for the attack-mask construction and re-orients per
side inside `_side_stats`. The side-to-move scalar plane 12 is the
only non-piece plane consulted.

CRTK metadata, source labels, verification flags, engine scores,
and principal variations are **not** consulted.

## Stop-gradient contract

The operator vector is fully differentiable. Trunk diagnostics
forwarded under the `trunk_<name>` prefix retain their original
gradient state; the `mechanism_energy` augmentation uses
`overload_operator_l2.detach()` so the trunk's own mechanism-energy
diagnostic stays causally pinned to the trunk pool norm.

The joint pool feature used in the delta head is **not** detached so
the delta can co-train the trunk's pool path.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_applied`` /
  ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution`` (gate * delta)
- ``overload_operator_mean`` / ``overload_operator_max`` /
  ``overload_operator_l2`` -- aggregate operator-vector statistics
- ``overload_us_mean`` / ``overload_us_peak`` / ``overload_them_mean``
  / ``overload_them_peak`` -- per-side target exposure summaries
- ``overload_pinned_share_us`` / ``overload_pinned_share_them``
- ``overload_defender_burden_us`` / ``overload_defender_burden_them``
- ``overload_criticality_us`` / ``overload_criticality_them``
- ``trunk_<name>`` for every diagnostic the i193 trunk produced
- ``mechanism_energy`` augmented with ``overload_operator_l2.detach()``
- ``proposal_profile_strength`` = ``|delta| * gate_entropy``

## Ablation modes

See ``ALLOWED_ABLATIONS``. The primary falsifier is
``no_cross_target_load``, which drops the cross-target overload
term `L^2 - Σ_t O^2` and leaves only single-target under-defence.
Rule-feature falsifiers are ``no_pins`` (π = 0 everywhere),
``no_target_value`` (uniform `v_tar, v_att, v_def`), and
``counts_only`` (drop `a_val, d_val, m_att, m_def` from the target
criticality gate). Numerical-recovery ablations are ``zero_delta``,
``trunk_only``, and ``disable_gate``.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` with a closed-form square-
centric overload builder (`DefenderOverloadBuilder`) that no
existing builder reproduces. It does not call
``build_research_packet_probe_from_config`` and does not delegate to
a shared CNN / MLP baseline builder.

## Numerical notes

- Target criticality goes through a `softplus` so `c(t) ≥ 0`.
- Both `defender_burden` and `target_exposure` are clamped non-
  negative; the `L^2 - Σ O^2` term is mathematically non-negative
  (since it equals `Σ_{t≠u} O O`, a sum of non-negative products),
  but rounding in mixed precision can produce tiny negatives.
- `NaN` and `inf` produced by all-empty target sets (e.g. the bare-
  kings position with no overloads anywhere) are sanitised at the
  side-vector boundary via `torch.nan_to_num(..., 0.0)`.
- The cheapest-attacker / cheapest-defender minima are computed
  with `masked_fill(inf).amin()` and then `where(isfinite)` so
  targets with no attacker / defender produce a zero entry.

## Geometry / orientation note

The simple_18 piece planes are *absolute* (planes 0..5 = white,
6..11 = black). The primitive re-orients to mover-perspective
inside `_side_stats`, so its event channels are colour-agnostic:
a white-to-move overload position and the colour-flipped black-to-
move overload position generate identical channel mass. This is the
same orientation contract as p049.

## Production upgrade path

The full attack builder is `einsum` + boolean masks; the overload
core is two BMMs and a small MLP. `torch.compile` should fuse the
dependent ops; this is the spec's preferred path. The phase-3
upgrade ("native i018 relation tensor integration") is out of scope
for this idea-folder; the source markdown lists it as the next step
after a positive standalone keep-decision.

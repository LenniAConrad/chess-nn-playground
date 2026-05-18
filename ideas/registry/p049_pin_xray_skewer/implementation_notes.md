# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/pin_xray_skewer.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Shared trunk helper: `src/chess_nn_playground/models/primitives/trunk_features.py`
  (`trunk_joint_features`).
- Idea-local wrapper: `ideas/registry/p049_pin_xray_skewer/model.py`.
- Registry key: `pin_xray_skewer`.
- Source primitive: `ideas/research/primitives/external_44_pin_xray_skewer_primitive.md`.

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
Mover-oriented piece planes are computed inline by swapping the white
and black halves using the side-to-move scalar plane 12. The
occupancy mask is the clamped sum of the 12 piece-presence planes.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The event tensors are fully differentiable. Trunk diagnostics fed
into the gate (`mechanism_energy`, etc.) are detached when forwarded.
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
- ``pxs_event_total_mean`` -- mean of all event channels per sample
- ``pxs_<event>_mean`` and ``pxs_<event>_max`` for
  `<event> in {xray1, abs_pin, rel_pin, discovered, skewer, pinned_defender}`
- ``trunk_<name>`` for every diagnostic the i193 trunk produced
- ``mechanism_energy`` augmented with ``pxs_event_total_mean.detach()``
- ``proposal_profile_strength`` = ``|delta| * gate_entropy``

## Ablation modes

See ``ALLOWED_ABLATIONS``. Primary falsifier is ``no_xray1``
(zeros every event term that uses `second_occ`). Rule-feature
falsifiers are ``uniform_values`` (no value context), ``no_pin_def``
(no defender-load channel), and ``shuffle_rays`` (rule-derived ray
geometry decoupled from the position).

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` with a fused event builder
(`PinXraySkewerBuilder`) that no existing builder reproduces. It does
not call ``build_research_packet_probe_from_config`` and does not
delegate to a shared CNN / MLP baseline builder.

## Numerical notes

- The per-piece-type value field is a `softmax` over a 6-dim logit
  vector. The softmax keeps values bounded in (0, 1) and sums to 1.0,
  so the worst-case event mass per source square is bounded above by
  the number of slider rays (8) times the value of the second
  occupant (<= 1), times the number of step positions (<= 7). This
  is a finite, position-independent upper bound.
- Per-event sigmoid scales start at 0.5 and are free to push toward
  0.0 (disable channel) or 1.0 (full weight).
- The `cumsum` is over the 7-step axis and runs in fp16 / fp32
  natively; no manual log-domain handling is needed because the
  values are small integers (0..7).

## Geometry / orientation note

The simple_18 piece planes are *absolute* (planes 0-5 = white,
6-11 = black). The primitive re-orients to mover-perspective inside
the forward, so its event channels are colour-agnostic: a pin event
from white-to-move and the colour-flipped pin event from
black-to-move generate the same channel mass.

## Production upgrade path

The full event builder is gather + cumsum + boolean masks + sum -- no
Python-side ray loops. `torch.compile` should fuse the dependent
ops; this is the spec's preferred path over a deferred Triton kernel.
Native i018 relation-tensor integration is the spec's deferred
"phase 3" and is out of scope for this idea-folder.

# Architecture

`Rule-Automorphism Quotient Bottleneck Network` (`RAQ-Net`, idea i048) is a
board-only `puzzle_binary` classifier whose central operator is the masked
Reynolds quotient over the safe chess automorphism groupoid
``G_x \subseteq {I, C, H, HC}`` defined in `math_thesis.md`. The
implementation replaces the shared `ResearchPacketProbe` mechanism profile
with materially distinct trainable code that follows the research packet
section "Architecture Specification" directly.

## Forward Pipeline

1. **`Simple18AutomorphismOrbit`.** A deterministic, parameter-free orbit
   adapter constructs ``X_orbit = (B, K_max=4, 18, 8, 8)`` plus the
   sample-wise validity mask ``orbit_mask = (B, 4)`` for the four-element
   transform list ``[I, C, H, HC]``:
   - ``I`` (identity) and ``C`` (color/turn rank-mirror with white/black
     piece-plane swap, side-to-move complement, ``KQkq <-> kqKQ`` castling
     swap, en-passant rank flip) are always valid.
   - ``H`` (file mirror) and ``HC`` (composition) are valid only when all
     four castling-rights planes are zero, satisfying the
     ``use_file_mirror_if_castling_absent`` rule.
   The adapter fails closed when the channel schema is anything other than
   `simple_18`/18 channels unless explicitly opted out via
   ``fail_closed_unknown_channels=False``. With ``pseudo_orbit=True`` the
   adapter emits the central same-count rank/file/rank-file flips that
   preserve view count and nuisance statistics but break color/side/castling
   semantics, exactly as required by the packet's central falsifier.
2. **Shared `SharedBoardEncoder`.** The orbit is flattened to
   ``(B*K_max, 18, 8, 8)`` and passed through a single shared CNN tower:
   a `Conv(18 -> hidden_channels)` stem with optional BatchNorm and GELU,
   followed by ``num_res_blocks`` ``ResidualBoardBlock`` units (two
   `Conv3x3 -> norm -> GELU` legs with residual + GELU), global average
   pooling, and a `Linear(hidden_channels, latent_dim)` projection with
   `LayerNorm + GELU`. Output: ``z_flat`` of shape
   ``(B*K_max, latent_dim)``.
3. **`MaskedReynoldsPool`.** Reshape to ``z = (B, K_max, latent_dim)`` and
   compute the masked Reynolds quotient
   ``z_bar = sum_k mask[:, k] * z[:, k] / sum_k mask[:, k]``. This is the
   exact masked orbit average from the math thesis.
4. **`QuotientClassifier`.** A `LayerNorm + Linear(latent_dim, 2)` head
   consumes ``z_bar`` to produce two-class quotient logits. The same head
   is broadcast over each orbit view to expose per-view logits used by the
   risk-variance penalty.
5. **`OrbitProjectionHead`.** A two-layer MLP
   (`latent_dim -> latent_dim -> projection_dim`) with `LayerNorm`, `GELU`,
   and dropout produces VICReg-style projections for each orbit view, used
   by the orbit-consistency, variance, and covariance terms.
6. **Primary logit collapse.** For ``num_classes=1`` (the puzzle-binary
   contract) the binary logit is the difference of the two-class logits
   ``two_class_logits[:, 1] - two_class_logits[:, 0]``, returned with
   shape ``(B,)``. For ``num_classes=2`` the head emits the two-class
   vector directly with shape ``(B, 2)``.

## Output Contract

`forward(x)` returns a `dict` keyed by:

- ``logits``: shape ``(B,)`` for ``num_classes=1`` (the single-logit BCE
  used by the puzzle-binary contract) or ``(B, 2)`` for ``num_classes=2``.
- ``two_class_logits``: ``(B, 2)`` quotient logits used internally to
  compute the binary collapse.
- ``valid_view_count``: per-sample ``|G_x|``, exposed for diagnostics and
  the count-stratified ablation report.
- ``file_mirror_valid``: per-sample float flag for whether ``H`` was a
  valid view, useful for the castling-absent stratification.
- ``orbit_variance`` / ``masked_orbit_variance``: latent variance around
  the Reynolds mean, the falsifier observable for orbit collapse.
- ``view_logit_variance``: per-sample variance of the per-view binary
  logits, the proxy for the REx-style transform-risk penalty
  ``L_risk``.
- ``symmetry_residual``: ``sqrt(view_logit_variance)``, a positive
  symmetry-residual scalar usable as a calibration diagnostic.
- ``orbit_consistency``: per-sample mean squared deviation of each valid
  projection vector from its within-orbit mean, the central
  ``L_orbit_inv`` observable used in the consistency objective.
- ``reynolds_norm`` / ``projection_norm`` / ``mechanism_energy``: latent
  magnitude diagnostics for monitoring training health and avoiding
  collapse.
- ``risk_variance_proxy``: alias for ``view_logit_variance`` carried as
  the named REx-style observable.

When called with ``return_aux=True`` the same forward additionally
exposes the raw orbit tensors and pre-aggregated VICReg/orbit losses
(``orbit_mask``, ``z``, ``projection``, ``view_logits``,
``view_two_class_logits``, ``orbit_consistency_loss``,
``vicreg_variance_loss``, ``vicreg_covariance_loss``,
``latent_small_loss``) so that a custom training path can wire the
``L_orbit + alpha_v * L_var + alpha_c * L_cov + lambda_rex * L_risk``
objective from the packet without re-running the encoder.

## Falsifier Wiring

The central same-count pseudo-orbit falsifier from the research packet
is wired directly into the same module class via ``pseudo_orbit=True``:
the adapter emits ``[I, rank_flip, file_flip, rank_file_flip]`` with all
views unconditionally valid, leaving encoder capacity, parameter count,
view count, and loss shape intact while breaking the color/side/castling
semantics. Setting ``use_file_mirror_if_castling_absent=False`` recovers
the color/turn-only ablation. ``use_color_turn_reversal=False`` reduces
the orbit to identity-only for the single-view augmentation control.

## Implementation Binding

- Registered model name: `rule_automorphism_quotient_bottleneck_network`
- Source implementation file: `src/chess_nn_playground/models/rule_automorphism_quotient.py`
- Idea-local wrapper: `ideas/registry/i048_rule_automorphism_quotient_bottleneck_network/model.py`

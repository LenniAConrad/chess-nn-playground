# Math Thesis

Source: `ideas/research/primitives/external_49_efficient_ray_occlusion_scan_primitive.md`
(single-proposal markdown; the file's working title is
``p048_efficient_ray_occlusion_scan_primitive``, but the queue assigns
``p054`` because p047..p053 are reserved by the same primitive batch
that promoted external_42..external_48).

## Working thesis

For a position with simple_18 board tensor and the fixed
queen-direction geometry `c_{s, d, l}` from
`ideas/research/primitives/external_49_*`:

1. Build a 16-channel per-square feature `feat[i] = (us_occ_i,
   them_occ_i, us_val_i, them_val_i, us_one_hot_i, them_one_hot_i)`
   from the simple_18 piece planes. Build an occupancy scalar
   `o_i = clamp(sum_p piece_plane_p(i), 0, 1)`.
2. Gather `o` and `feat` along all 8 queen directions and 7 steps,
   producing `occ_ray in R^{B x 8 x 64 x 7}` and `feat_ray in R^{B x 8
   x 64 x 7 x 16}`. Off-board steps are zeroed by `step_mask`.
3. Compute the inclusive blocker prefix
   `k_{b, s, d, l} = sum_{q <= l} occ_ray_{b, s, d, q}` (one
   `torch.cumsum` call along the step axis).
4. Derive four hard 0/1 masks (`visible`, `first`, `second`,
   `xray_lane`) by equality tests on `k` and `k - occ_ray`:

       visible_{b, s, d, l}   = 1[k - o == 0] * step_mask
       first_{b, s, d, l}     = o * 1[k == 1] * step_mask
       second_{b, s, d, l}    = o * 1[k == 2] * step_mask
       xray_lane_{b, s, d, l} = 1[k - o == 1] * step_mask

5. Compute the first / second blocker feature summaries by masked
   reductions over the gathered feature tensor:

       first_feat_{b, s, d}  = sum_l first_{b, s, d, l} * feat_ray_{b, s, d, l}
       second_feat_{b, s, d} = sum_l second_{b, s, d, l} * feat_ray_{b, s, d, l}

   And the 13 per-source-per-direction summaries that drive the head:
   `visible_count`, `mobility_len`, `xray_lane_len`,
   `first_exists`, `first_value`, `first_us_occ`, `first_them_occ`,
   `second_exists`, `second_value`, `second_us_occ`, `second_them_occ`,
   `xray_pressure`, and `discovered_pressure + pinned_to_king`.
6. Project to per-square (rook-line, bishop-line, queen-line) using
   the direction-class masks `ortho_mask = (1, 0, 1, 0, 1, 0, 1, 0)`
   and `diag_mask = (0, 1, 0, 1, 0, 1, 0, 1)`. Mean + max pool over
   the 64 squares yields the head readout vector.
7. Delta head: `primitive_delta_raw = MLP(cat(readout, joint))`. Gate
   head: `primitive_gate = sigmoid(MLP(cat(joint, occ_density,
   mobility_mean, xray_mean)))`. Output:

       final_logit = base_logit + primitive_gate * primitive_delta_raw.

## Why this matters

Sliding-piece attacks depend on occupancy along a direction, not on
local convolutions and not on ray density alone. X-rays are about a
piece controlling through an intervening piece, and discovered attacks
are about one piece moving away to reveal another. Those phenomena
require the primitive to know the **first blocker**, the **second
blocker**, their **side**, and the **value of the target**, not just
whether a line is "open enough".

The current i018 visibility builder contracts a dense
`(64, 64, 64)` source-target-between cube against occupancy
(`262144` slots per board before features). The compact scan works
over the legal ray representation directly (`3584` padded slots,
`1456` valid cells per board). p020 and p021 already touch the same
geometry but collapse the ray (p020 via a recurrence, p021 via a
transmittance product) instead of preserving first / second blocker
identity in one fused pass.

## What is actually proven

- The `cumsum` over `occ_ray` is exactly the inclusive blocker count
  along each ray; the four hard masks are mechanical consequences of
  the equality tests.
- The masked reductions `first_feat = sum_l first * feat_ray` and
  `second_feat = sum_l second * feat_ray` recover the exact 16-channel
  feature at the first / second occupied square on the ray (because
  `first` and `second` are one-hot along the step axis whenever the
  ray contains at least one / two blockers).
- The compact tensor layout `(B, D, S, L)` has padded size `3584` per
  board versus `262144` for the `(64, 64, 64)` cube, so the working
  set is ~73x smaller (asymptotic argument from the source markdown).
  The realised speedup depends on `gather + cumsum` fusion on the
  target GPU and must be measured -- see ``ablations.md``.
- The hybrid gate-and-delta scaffold is identical in shape to p020 /
  p021 / p046, so the additive logit delta contract is preserved.

## What is only hypothesized

- That preserving first and second blocker identity (and the
  derived discovered / pin candidates) carries discriminative chess
  signal not already encoded by the i193 trunk and its conv layers.
- That the compact scan beats the looped p020 / p026 implementations
  *and* the i018 dense visibility builder in steady-state GPU timing
  (asymptotic argument is favorable, but `gather + cumsum` fusion
  under `torch.compile` is the load-bearing detail).
- That the dense-edge scatter mode described in the research markdown
  (per-source ``(B, 64, 64)`` `rook_visible` / `bishop_xray` etc.) is
  the right interface for an i018 graph-builder replacement. The
  current implementation does *not* expose dense edges; they remain
  deferred until the i018 integration target is in scope.

## Failure cases

1. *First blocker is enough*: tested by `first_only` (drop all
   second-blocker / xray / discovered / pin channels). If the
   unablated head matches `first_only`, the second-blocker structure
   is not load-bearing.
2. *Blocker identity not needed*: tested by `no_blocker_id` (zero
   only the side / value channels). If the unablated head matches
   `no_blocker_id`, the operator is doing geometry rather than
   tactical content.
3. *Mask irrelevant*: tested by `uniform_occupancy`,
   `empty_occupancy`, `shuffle_occupancy`.
4. *Gate carries everything*: tested by `disable_gate`.
5. *Head is dead weight*: tested by `zero_delta` / `trunk_only`.

## Falsifier

- `first_only` -- primary. Strips second-blocker / x-ray / discovered
  / pin channels. The unablated head must beat this control on the
  target slice (positions whose label depends on what sits behind a
  single blocker).
- `no_blocker_id` -- secondary. Zeros the side / value identity
  channels but keeps geometry. Tests whether the operator is just
  reading mobility geometry.
- `uniform_occupancy` -- mask-irrelevance control.
- `shuffle_occupancy` -- decouples mask from position.
- `empty_occupancy` -- pure-geometry control.

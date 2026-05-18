# Math Thesis

Source: `ideas/research/primitives/external_44_pin_xray_skewer_primitive.md`.

## Working thesis

For a position with `simple_18` board tensor
`x in {0, 1}^{B x 18 x 8 x 8}`:

1. Re-orient piece planes to mover-perspective. Let `us in {0, 1}^{B x 6 x 64}`
   be our pieces (P, N, B, R, Q, K) and `them in {0, 1}^{B x 6 x 64}` the
   opponent's, selected by the side-to-move scalar `stm = mean(x[12])`.
2. Build the per-direction slider activation
   `S_f(s) = 1_{f matches direction d}` for each source square `s`:

       S(s, d) = us_Q(s) + us_R(s) * 1[d orth] + us_B(s) * 1[d diag].

3. Along each direction `d` from each source square `s`, walk 7 ray
   steps using the precomputed `(8, 64, 7)` ray-index table. Gather
   occupancy `o[d, s, l] in {0, 1}` and per-piece indicators along
   the ray. Compute the cumulative occupancy `c[d, s, l] = sum_{j<=l} o[d, s, j]`
   and the masks

       first_occ[d, s, l] = o[d, s, l] * 1[c[d, s, l] = 1],
       second_occ[d, s, l] = o[d, s, l] * 1[c[d, s, l] = 2].

   The chess invariant `first_occ` is the unique blocker on the ray;
   `second_occ` is the next occupant *behind* it. This is the
   tensorised form of standard "first / second occupied square"
   bitboard scans.
4. Let `v_them(t) = sum_p w_p * them_p(t)` be the per-square enemy-
   value field with `w = softmax(W)` over 6 piece-type logits. The
   same definition applies to `v_us(t)`. Pre-default `W` is set to
   `(1, 3, 3, 5, 9, 12)` so the softmax mass concentrates on the
   king.
5. Form per-ray event scalars by step-summing the masked sequences:

       first_them_per_ray = sum_l first_occ * them_any,
       second_king_per_ray = sum_l second_occ * them_king,
       second_value_per_ray = sum_l second_occ * v_them,    etc.

6. The six event masses are

       xray1(s, d)     = S(s, d) * (first_them + first_us) * second_value
       abs_pin(s, d)   = S(s, d) * first_them * second_king
       rel_pin(s, d)   = S(s, d) * first_them * (second_queen + 0.6 second_rook)
       discovered(s, d) = S(s, d) * first_us * second_value
       skewer(s, d)    = S(s, d) * second_any * relu(first_value - second_value)
       pinned_def(s, d) = S(s, d) * first_value * second_king.

7. Direction-sum each event to `(B, 6, 64)` per-square channels,
   apply a per-event sigmoid scale, mean/max-pool to a 12-dim
   summary, and feed `cat(joint_pool, summary)` to the delta head.

## Why this matters

Generic ray heads (p020 SSM, p021 semiring scan, p034 selective scan)
mix line content into a single hidden state and rely on the trunk to
unpack which-piece-first / which-piece-second. PXS exposes that
structure directly as typed channels, weighted by piece-type value.
The `pin` and `skewer` slices specifically require the model to
distinguish "first occupant is enemy" from "first occupant is friend",
which a hidden-state scan cannot do without burning capacity. The
source spec also notes that p020 and p034 explicitly document
Python-side scan loops as their remaining speed problem; PXS avoids
that by construction.

## What is actually proven

- The cumsum-based `first_occ` / `second_occ` masks match the
  bitboard "first / second occupied square" semantics exactly on
  well-formed `(8, 64, 7)` ray sequences. The unit test
  `test_abs_pin_event_fires_on_rook_pawn_king_axis` constructs an
  explicit rook-pawn-king pin and verifies `abs_pin` fires only at
  the slider source.
- Each event is non-negative, so the per-ray sum is well-defined.
- The `zero_delta` ablation recovers the trunk's `base_logit`
  bit-for-bit (verified by
  `test_zero_delta_recovers_trunk_logit`).
- The `shuffle_rays` ablation is computed exactly the same way as
  the unablated path, just with a permuted index buffer, so any
  finite difference observed in training would be attributable to
  the rule-derived geometry rather than to numerical noise.
- The gradient flows through the gate, the delta MLP, the event
  scales, and the trunk via the joint feature (verified by
  `test_backward_gradients_flow_through_head_and_trunk`).

## What is only hypothesised

- That the typed event channels carry chess-specific information
  beyond what i193 already encodes spatially. The four falsifiers
  below test this.
- That the simple per-source-square defender-load proxy
  (`first_value * second_king`) is a meaningful substitute for the
  full `D_def(b) = sum_u A_same(b, u) v(u)` form in the spec. The
  full form would require building a same-side defence graph, which
  is out of scope for a primitive head.

## Failure cases

1. *Pinned-attacker inflation*: the spec calls out that raw attack
   maps can mis-count pinned pieces as attackers. PXS does **not**
   compute attack maps; it only computes ordered occupant masks. So
   this failure mode does not apply.
2. *Coefficient explosion*: piece-value logits are `softmax`-bounded
   into (0, 1); per-event scales are `sigmoid`-bounded into (0, 1).
3. *Two-or-more relevant blockers on the same line*: by construction
   PXS only looks at the first and second occupied squares per ray.
   The third occupant and beyond are invisible. This is the spec's
   declared weakness; see ablations.md.
4. *Order-scramble*: tested by `shuffle_rays`.

## Falsifier

- `no_xray1` -- primary. Zero every event term that depends on
  `second_occ`. If `pin` / `skewer` / `discovered_attack` slice
  lift survives, the operator was not using one-blocker x-ray logic.
- `uniform_values` -- replace the per-piece-type value softmax with
  uniform `1/6`. Tests value context.
- `no_pin_def` -- zero the pinned-defender channel. Tests defender-
  load load-bearing.
- `shuffle_rays` -- permute the rule-derived ray geometry.

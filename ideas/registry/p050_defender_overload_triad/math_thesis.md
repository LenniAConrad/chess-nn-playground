# Math Thesis

Source: `ideas/research/primitives/external_45_defender_overload_triad_primitive.md`.

## Working thesis

For a position with `simple_18` board tensor
`x ∈ {0, 1}^{B × 18 × 8 × 8}`:

1. Extract absolute piece planes `ps ∈ {0, 1}^{B × 12 × 64}` (white
   planes 0..5, black planes 6..11) and the side-to-move scalar
   `stm = mean(x[12]) ∈ {0, 1}`. Occupancy is
   `occ = clamp(ps.sum(dim=1), 0, 1)`.
2. Build rule-derived attack tables `geom_attacks ∈ {0,1}^{6×2×64×64}`
   (per piece type × colour × source × target) and the slider-blocker
   `between ∈ {0,1}^{64×64×64}`. Per-colour attack maps
   `attack[colour] ∈ {0,1}^{B×64×64}` are produced by
   `Σ_piece source_planes[piece, colour] ⊗ geom_attacks[piece, colour]`,
   slider rays gated by
   `clear[b, s, t] = 1[ between[s, t, :] · occ[b, :] ≤ 0.5 ]`.
3. Side-rotate to mover perspective with stm:

       us_attack = stm · attack[W] + (1 - stm) · attack[B]
       them_attack = stm · attack[B] + (1 - stm) · attack[W]
       us_any, them_any, us_value, them_value, us_king, them_king

   defined analogously from the per-side piece planes.
4. Cumsum pin detector. For each side σ, build the
   per-direction enemy-slider field
   `enemy_sliders[σ][d, s] = 1 iff a slider firing along d sits at s`
   (queen always, rook for orthogonal d, bishop for diagonal d).
   Gather occupancy and own-piece indicators along the
   `ray_geometry (8, 64, 7)` rays. The first/second occupant masks
   `first_occ`, `second_occ` are exactly the cumsum-1 / cumsum-2
   patterns shared with p049. The pin indicator
   `πσ ∈ {0, 1}^{B × 64}` is then
   `scatter_add( first_own · 1[second is enemy slider firing along d],
   target_squares )` restricted to king-sourced rays.
5. **Side stats** (per side σ ∈ {us, them}). Restrict attack and
   defense to enemy-occupied targets
   `attack ⊙ target_piece.unsqueeze(1)`,
   `defense ⊙ target_piece.unsqueeze(1)`. Compute per-target
   features

       a(t)     = Σ_i attack[i, t]            attack count per target
       d(t)     = Σ_d defense[d, t]           defense count per target
       p(t)     = Σ_d defense[d, t] · π(d)    pinned-defender count per target
       a_val(t) = Σ_i attack[i, t]  · v_att(i)
       d_val(t) = Σ_d defense[d, t] · (1 - λ·π(d)) · v_def(d)
       m_att(t) = min_{i : attack[i, t]} v_att(i)
       m_def(t) = min_{d : defense[d, t] · (1 - π(d))} v_def(d)
       x(t)     = [a, d, p, a_val, d_val, m_att, m_def, v_tar]
       c(t)     = softplus(gθ(x(t))) · 1[target_piece(t) > 0]

   `gθ` is a tiny `LayerNorm + Linear + GELU + Linear` MLP. Then form

       O(d, t) = defense[d, t] · c(t)
       L(d)    = Σ_t O(d, t)
       m(d)    = 1 + μ · π(d)

   and the closed-form overload masses

       Ω_def(d) = m(d) · (L(d)^2 - Σ_t O(d, t)^2)
       X_tar(t) = c(t) · [ defense^T (m·L) - c · (defense^2)^T m ](t).

6. Pool to a 5-feature side vector

       S_σ = [
         mean_t X_tar(t),
         max_t X_tar(t),
         mean_{d : defender_occ(d)>0} Ω_def(d),
         pinned_share(σ),
         mean_{t : target_piece(t)>0} c(t),
       ]

   and concatenate
   `F = [S_us, S_them, S_us - S_them, |S_us - S_them|]` for a 20-dim
   operator vector. `pinned_share(σ) = Σ_t X_tar(t) · (p / d.clamp_min(1.0))
   / Σ_t X_tar(t)` is the share of overload exposure mediated by
   pinned defenders.

7. Feed `cat(joint_pool, operator_vector)` to the delta head and
   `cat(joint_pool, |operator| mean)` to the gate head; combine
   `final_logit = base_logit + sigmoid(gate) · delta_raw`.

## Why this matters

The key algebraic identity is

    L(d)^2 - Σ_t O(d, t)^2 = Σ_{t ≠ u} O(d, t) · O(d, u).

So `Ω_def(d)` is exactly the *weighted mass of distinct critical
targets simultaneously assigned to defender d*. Plain per-target
attack/defense counts cannot distinguish "two defenders, one target
each" from "one defender, two targets" because both produce the same
marginal `a, d` histogram. The cross-product term resolves that
identity ambiguity in `O(BN^2)` -- no `(B, N, N, N)` triple is
materialised.

## What is actually proven

- `Ω_def(d)` matches the cross-target reuse mass
  `Σ_{t≠u} O(d, t) · O(d, u)` algebraically -- this is a direct
  expansion of the square.
- The cumsum-based first/second occupant masks match standard
  bitboard "first / second occupied square" semantics (the same fact
  used by p049).
- The `zero_delta` ablation recovers the trunk's `base_logit`
  bit-for-bit (verified by `test_zero_delta_recovers_trunk_logit`).
- The `no_cross_target_load` ablation strictly removes the overload-
  proper signal, leaving only single-target under-defence (verified
  numerically by the unit-test position with a single defender
  covering two attacked targets).
- Gradients flow through the trunk, the target-criticality MLP, the
  piece-value field, the pin parameters, the delta head, and the
  gate head (verified by
  `test_backward_gradients_flow_through_head_and_trunk`).
- An explicit overload position (white queen attacks two black
  pawns, both defended only by one black knight) yields strictly
  greater `defender_burden_max` than an equivalent position without
  the shared defender (verified by
  `test_overload_signal_higher_on_real_overload_position`).
- A position with an absolute rook-pawn-king pin produces
  `πthem(a4) = 1.0` (verified by
  `test_pin_indicator_fires_on_absolute_pin`).

## What is only hypothesised

- That defender-identity reuse carries chess-specific information
  beyond what the i193 trunk already encodes spatially. The four
  surgical ablations (`no_cross_target_load`, `no_pins`,
  `no_target_value`, `counts_only`) test this.
- That the simple criticality MLP is a meaningful substitute for a
  full SEE / minimax exchange evaluator. The spec calls this a
  "SEE-light" compromise rather than a full recursive SEE.

## Failure cases

1. *Cubic combinatorial explosion*. The closed-form rewrite avoids
   the `(B, N, N, N)` triple entirely; cost stays `O(BN^2)`.
2. *Pinned-attacker inflation*: attacker masks count pinned pieces
   as attackers, but the matched falsifier `no_pins` and the
   pin-discounted `d_val` directly probe this.
3. *King-value swamping*: the default piece-value vector is
   `(1, 3.2, 3.3, 5, 9, 9)` -- the king is clipped to queen-level so
   ordinary occupied-target overload is not swamped by an
   indistinguishable king-on-target term.
4. *Endgame zugzwang positions*: low attack/defence incidence will
   reduce all targets to near-zero criticality, so the operator
   will simply produce a near-zero delta. The gate is initialised
   near-closed so the i193 baseline is preserved in that regime.

## Falsifier

- `no_cross_target_load` -- primary. Drops the `L^2 - Σ O^2` term;
  leaves only single-target under-defence. If slice lift survives,
  the operator is not actually measuring defender reuse.
- `no_pins` -- secondary. Sets `π = 0` everywhere. Tests pin load-
  bearingness.
- `no_target_value` -- secondary. Sets `v_tar, v_att, v_def = 1`.
  Tests piece-value weighting.
- `counts_only` -- tertiary. Drops `a_val, d_val, m_att, m_def`
  from the target-criticality gate. Tests SEE-light load-bearingness.

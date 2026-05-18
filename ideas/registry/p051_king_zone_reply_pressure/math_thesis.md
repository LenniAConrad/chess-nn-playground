# Math Thesis

Source: `ideas/research/primitives/external_46_king_zone_reply_pressure_primitive.md`.

## Working thesis

For a position with `simple_18` board tensor
`x ∈ {0, 1}^{B × 18 × 8 × 8}`:

1. Extract absolute piece planes `ps ∈ {0, 1}^{B × 12 × 64}` (white
   planes 0..5, black planes 6..11) and the side-to-move scalar
   `stm = mean(x[12]) ∈ {0, 1}`. Occupancy is
   `occ = clamp(ps.sum(dim=1), 0, 1)`.
2. Build rule-derived attack tables `geom_attacks ∈ {0,1}^{6×2×64×64}`
   (per piece type × colour × source × target) and the slider-blocker
   `between ∈ {0,1}^{64×64×64}`. Per-colour attack maps are produced
   as
   `attack[colour] ∈ R^{B×64}`
   with
   `attack_σ_nom(t) = Σ_{p,s} u(p) · src_p_σ(s) · geom_attacks[p, σ, s, t] · clear[b, s, t]`
   and
   `attack_σ_free(t) = Σ_{p,s} u(p) · src_p_σ(s) · (1 - λ_pin · π_σ(s)) · geom_attacks[p, σ, s, t] · clear[b, s, t]`,
   where `clear[b, s, t] = 1[ between[s, t, :] · occ[b, :] ≤ 0.5 ]`,
   `u(p)` are softplus-bounded learnable attack units, and `λ_pin` is
   the sigmoid-bounded pin discount.
3. Side-rotate to mover perspective with `stm`:

       us_attack    = stm · attack_w_nom  + (1-stm) · attack_b_nom
       them_attack  = stm · attack_b_nom  + (1-stm) · attack_w_nom
       us_def_free  = stm · attack_w_free + (1-stm) · attack_b_free
       them_def_free = stm · attack_b_free + (1-stm) · attack_w_free

   and analogous for nominal defender masses. Defender-king selectors
   are
   `them_king = stm · black_king_sq + (1-stm) · white_king_sq`,
   `us_king` analogously.

4. Cumsum pin detector. For each side σ, build the per-direction
   enemy-slider field
   `enemy_sliders_σ[d, s] = 1 iff an enemy slider firing along d
   sits at s` (queen always, rook for orthogonal d, bishop for
   diagonal d). Gather occupancy and own-piece indicators along the
   `ray_geometry (8, 64, 7)` rays from the OWN king. The first /
   second occupant masks `first_occ`, `second_occ` are the cumsum-1 /
   cumsum-2 patterns shared with p049 / p050. The pin indicator
   `π_σ ∈ {0, 1}^{B × 64}` is then
   `scatter_add( first_own · 1[second is enemy slider firing along d],
   target_squares )` restricted to king-sourced rays.

5. King-zone masks. Precomputed `ring_mask (64, 64)` is colour-
   agnostic: `ring_mask[k, q] = 1[ q ∈ N_8(k) and q ≠ k ]`. The
   forward masks are colour-conditional:
   `front_mask_w[k, q] = 1[ row(q) = row(k) - 2 and |file(q) - file(k)| ≤ 1 ]`
   for white-king defenders and
   `front_mask_b[k, q] = 1[ row(q) = row(k) + 2 and |file(q) - file(k)| ≤ 1 ]`
   for black-king defenders. In `simple_18` row indexing row 0 is
   rank 8, so the attacker direction is toward decreasing row for
   white defenders and toward increasing row for black defenders.

6. **Side vector** (per side σ ∈ {us, them}). Let `K`, `A`, `D_nom`,
   `D_free`, `D_occ`, `T_occ` denote the defender-king one-hot,
   attacker mass, nominal / free defender mass, defender-side
   occupancy, and total board occupancy (all `(B, 64)`).

   * **Zone pressure.** With softplus-bounded weights
     `(w_K, w_e, w_o) ≈ (4, 3, 2)`, learnable `λ_def ∈ (0, 1)` and
     softplus `η`,

           net = max(0, A - λ_def · D_free)
           ZP_core = Σ_q ( w_K · K(q) + w_e · ring(q) · (1 - T_occ(q))
                          + w_o · ring(q) · T_occ(q) ) · net(q)
           ZP_front = η · Σ_q front(q) · net(q)
           zone_pressure = ZP_core + ZP_front.

   * **Fake-defense loss.**

           fd_loss = Σ_{q ∈ K ∪ ring ∪ front} max(0, D_nom(q) - D_free(q)).

   * **Escape decomposition.** With
     `empty(q) = 1[ T_occ(q) ≤ 0.5 ]` and
     `attacked(q) = 1[ A(q) > ε ]`,

           live(σ)    = Σ_q ring(q) · empty(q) · (1 - attacked(q))
           sealed(σ)  = Σ_q ring(q) · empty(q) · attacked(q)
           blocked(σ) = Σ_q ring(q) · D_occ(q).

   * **Current check severity.**

           king_attack_mass(σ) = Σ_q K(q) · A(q) = A(king_sq).

   * **Front-zone net pressure** (raw, before η).

           front_attack_mass(σ) = Σ_q front(q) · net(q).

   * **Reply proxy.** With sigmoid escape weights
     `(α1, α2, α3) ∈ (0, 1)^3` and
     `ring_free_def(σ) = Σ_q ring(q) · D_free(q)`,

           reply_proxy(σ) =
               log(1 + live + α1 · sealed + α2 · blocked
                     + α3 · ring_free_def).

   The side vector is

       S_σ = [zone_pressure, fd_loss, live, sealed, blocked,
              king_attack_mass, front_attack_mass, reply_proxy] ∈ R^8.

7. **Operator vector.** Concatenate
   `F = [S_us, S_them, S_us - S_them, |S_us - S_them|] ∈ R^32`.

8. Feed `cat(joint_pool, F)` to the delta head and
   `cat(joint_pool, mean |F|)` to the gate head; combine
   `final_logit = base_logit + sigmoid(gate) · delta_raw`.

## Why this matters

King safety is not a flat count. The five interpretable terms each
isolate a different axis the trunk does not see directly:

- `zone_pressure` is the standard king-safety pattern (weighted
  attack minus pin-discounted defense, with extra weight on the king
  square itself and on empty flight squares). The CPW-style attack
  units `(P=1, N=2, B=2, R=3, Q=5)` are the established king-safety
  prior.
- `fd_loss` separates *nominal* from *free* defense. A defender
  pinned to its own king cannot legally support the relevant
  interposition / capture without abandoning the king, so its
  contribution to `D_nom` is illusory.
- `live / sealed / blocked` partition the immediate king-flight
  options. Only `live` squares are usable flight; `sealed` are
  empty but already attacked; `blocked` are occupied by own
  pieces. Under double check only `live` matters; under single
  check `blocked` still contributes via interposition.
- `king_attack_mass` is a continuous proxy for "is the king in
  check, and how heavily." A non-zero value implies at least one
  legal-move family is forced to escape, capture, or interpose.
- `reply_proxy` is a cheap upper bound on the size of the
  defender's legal reply family without enumerating moves.
- The side-to-move asymmetry `S_us - S_them` is what separates
  "the side to move has a forcing attack" from "mutual king danger
  with equal reply capacity". This is conceptually the spec's
  `KZRP_Δ`.

## What is actually proven

- The pin indicator matches absolute-pin semantics for a single-
  blocker ray (verified by `test_pin_indicator_fires_on_absolute_pin`,
  shared with p049 / p050).
- The `zero_delta` ablation recovers the trunk's `base_logit` bit-
  for-bit (verified by `test_zero_delta_recovers_trunk_logit`).
- The `no_front_zone` ablation strictly removes the `η · Σ_{q ∈ Z_front}
  net` contribution from `zone_pressure` (verified numerically by
  building a position whose attack mass concentrates in the front
  squares).
- `no_pins` zeroes the pin indicator everywhere, which collapses
  `D_free` onto `D_nom` and therefore `fd_loss` to zero (verified
  numerically on a rook-pawn-king pin defending a king-ring square).
- The operator output is finite for bare-kings, opening, and
  middlegame positions (verified across the ablation matrix).
- Gradients flow through the trunk, the attack-unit field, the
  pin discount, the defense discount, the front strength, the zone
  weights, the escape weights, the delta head and the gate head
  (verified by `test_backward_gradients_flow_through_head_and_trunk`).

## What is only hypothesised

- That an explicit pin-discounted zone-pressure + escape +
  reply-capacity decomposition lifts the `mate_in_1` and near-
  puzzle slices over the i193 trunk by enough to clear the keep
  bar. The seven surgical ablations test the load-bearingness of
  each subterm.
- That the simple log reply-capacity proxy (live + α-sealed +
  α-blocked + α-ring-free-defense) is a meaningful surrogate for
  the full check-evasion reply family (king moves + captures of
  checkers + slider interpositions). The spec calls this a "cheap
  upper bound" rather than a full reply enumerator.
- That `Z_front` (one rank further than the king ring in the
  attacker direction) is the right scope. The spec lists "two or
  three additional forward squares" as the CPW convention; the
  implementation picks three for simplicity.

## Failure cases

1. *Endgame zugzwang positions*: low attack incidence collapses
   `net` to zero on the zone, so the operator produces a near-zero
   delta. The gate is initialised near-closed so the i193 baseline
   is preserved in that regime.
2. *Positions where the attacker is itself in check*: the operator
   computes both sides symmetrically, but `kzrp_asym_score` then
   reflects the mutual danger correctly.
3. *Material-imbalance positions where king safety is irrelevant*:
   the gate can learn to stay near zero (no contribution).
4. *Pinned attackers*: the spec does not discount attacker mass,
   only defender mass. Pinned attackers can still apply pressure
   along their pin axis, which matches chess semantics.

## Falsifier

- `no_front_zone` -- primary. Drops the `η · Σ_{q ∈ Z_front} net`
  term; if lift survives, the front-rank extension is not load-
  bearing and the operator collapses to a ring-only scalar.
- `no_pins` -- secondary. Sets `π = 0` everywhere; collapses
  `D_free = D_nom`. Tests pin / fake-defense load-bearingness.
- `uniform_zone_weights` -- secondary. Replaces `(4, 3, 2)` weights
  with uniform 1. Tests whether unequal king-square emphasis matters.
- `no_escape_decomp` -- secondary. Collapses live / sealed /
  blocked into a single total. Tests whether the decomposition
  is load-bearing or whether a single mobility scalar suffices.
- `uniform_units` -- tertiary. Sets all attack units to 1. Tests
  CPW-style attack-unit weighting.
- `no_asymmetry` -- tertiary. Sets `S_them = 0`. Tests whether the
  side-to-move asymmetry term is load-bearing.

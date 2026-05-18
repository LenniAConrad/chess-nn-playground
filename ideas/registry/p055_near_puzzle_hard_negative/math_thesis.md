# Math Thesis

Source: `ideas/research/primitives/external_50_near_puzzle_hard_negative_primitive.md`.

## Working thesis

For a position with `simple_18` board tensor:

1. Compute the i193 trunk spatial features and joint pool.
2. Pool `num_candidates` candidate tokens `t_m` and `num_replies`
   defender-reply tokens `t_r` via two learned `BoardTokenAttention`
   modules over the spatial map.
3. Per-candidate surface and verified scores:

       u_surf(m) = MLP_surf(t_m)
       u_ver(m)  = MLP_ver(t_m)

   The legality discount is `Disc(m) = u_surf(m) - u_ver(m)`.
4. Candidate-set softmax with temperature `tau_c`:

       pi(m) = softmax(u_ver(m) / tau_c)
       Conc  = 1 - H(pi) / log(num_candidates)
       Gap12 = softplus(u_ver_(1) - u_ver_(2))

5. Per-(candidate, reply) bilinear neutralization:

       n(m, r) = Bilinear(t_m, t_r)

   Aggregate:

       ReplyMass(m) = tau_r * logsumexp(n(m, r) / tau_r)
       SafeCount(m) = sum_r sigmoid(4 * (n(m, r) - safe_threshold))
       FG(m)        = u_ver(m) - ReplyMass(m)
       FG*          = max_m FG(m)
       m*           = argmax_m u_ver(m)
       FG(m*)       = FG[m*]
       Avail        = log(1 + SafeCount(m*))

6. Reply-channel information (RCI) — mutual information of the joint
   softmax over `n(m, r) / tau_r`:

       p(m, r) = softmax(n(m, r) / tau_r)          (flattened across m, r)
       MI      = sum_{m, r} p(m, r) * log(p(m, r) / (p(m) * p(r)))
       RCI     = MI / log(num_candidates)              clipped to [0, 1]

7. Bounded board-only signals derived from `simple_18`:

       defender_zone = max_pool(defender_king_plane, kernel = 2*r+1)
       attacker_zone = max_pool(attacker_king_plane, kernel = 2*r+1)
       KEP           = bounded ratio of zone attack-defense imbalance
       DOA           = bounded asymmetry of the two zone overloads
       d_bal         = scaled (surface attack - 0.5 * surface defend)
       Counter       = 1 - surface_signal

8. Diagnostic vector:

       z(x) = [FG*, FG(m*), Disc(m*), Conc, Gap12, Avail, RCI,
               d_bal, KEP, DOA, Counter]

9. Veto head:

       veto(x) = softplus(MLP(LayerNorm([z(x); joint])))

   The primitive contribution is `gate * (-veto(x))` so high veto
   pressure lowers the puzzle logit on the puzzle side of the
   threshold. The gate is initialized near closed
   (`gate_init = -2.0`) so the primitive starts as a small additive
   correction.

## Why this matters

Near-puzzle false positives are exactly the cases where the surface
tactical pattern looks like a puzzle but a legal reply refutes it.
A primitive that pairs surface and verified scores per candidate and
adds a soft-existential reply-neutralization aggregator can score
"there is a tempting move but a safe reply exists" without needing
fine labels, CRTK tags, engine evaluations, or principal variations.

## What is actually proven

- The candidate and reply pools are differentiable; the bilinear
  neutralization head is differentiable; `logsumexp` and `softmax`
  reductions are differentiable.
- The veto contribution is *sign-correct*: `softplus(.) >= 0` and the
  contribution is `-gate * softplus(.) <= 0`, so the primitive can
  only lower the puzzle logit (it cannot push borderline negatives
  *up* into the puzzle class).
- The `no_replies` and `no_legality_discount` ablations are
  semantically meaningful falsifiers: zeroing those `z` entries makes
  the corresponding load-bearing claim untestable.
- The `concentration_only` ablation reduces the head to a candidate-
  count concentration head.

## What is only hypothesized

- That a board-only candidate/reply attention pool can stand in for
  `python-chess` legal-move generation well enough to expose
  near-puzzle FP signal that survives both the `no_replies` and
  `no_legality_discount` ablations.
- That the bounded board-only `KEP`/`DOA` reductions add signal that
  is not already captured by the i193 trunk's spatial features.

## Failure cases

1. *Generic-tactic booster (not a rejector).* Tested by inspecting
   the sign and magnitude of the contribution on near-puzzle vs true
   puzzle samples; if veto fires symmetrically the rejection story is
   wrong.
2. *Replies are noise.* Tested by `no_replies` and `shuffle_replies`.
3. *Discount is noise.* Tested by `no_legality_discount`.
4. *Concentration alone suffices.* Tested by `concentration_only`.
5. *King-zone reductions add nothing.* Tested by `no_king_escape` and
   `no_overload`.

## Falsifier

- `no_replies` — primary. Forces `Avail = ReplyMass = RCI = 0`.
- `no_legality_discount` — primary. Forces `Disc(m*) = 0`.
- `concentration_only` — keeps only `Conc` and `Gap12`.
- `shuffle_replies` — in-batch permutation of reply tokens.
- `no_overload` / `no_king_escape` — single-feature drops.
- `zero_delta` / `trunk_only` — i193 baseline recovery.
- `disable_gate` — pin gate at 1.0 to test gate load-bearing.

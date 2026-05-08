# Math Thesis

Bounded Board Hinge Logic

Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0859_tuesday_new_york_bounded_hinge.md`.

Working thesis: Select **Bounded Board Hinge Logic**: a differentiable
logic classifier that compiles a fixed library of typed, shallow
PSL-style formulas into tensor operations over the current-board
predicate algebra and decides puzzle / non-puzzle by the energy gap of
a hinge loss between the two target classes.

## Setting

Encode the current board as a finite typed structure whose universe is
`V = {1, ..., 64}` and whose deterministic closed-world facts are the
unary predicates `C(x) in [0, 1]` and binary relations
`R(x, y) in [0, 1]` enumerated below. The model treats the side-to-move
as `us` (`u`) and the opponent as `them` (`t`), so every fact is
defined relative to the side selector. The fact extractor materialises

- **32 unary predicates**: piece occupancy and identity for each side
  (`u_pawn ... u_queen`, `t_pawn ... t_queen`, plus `u_king`, `t_king`,
  `u_piece`, `t_piece`, `empty`, `u_slider`, `t_slider`), static board
  attributes (`center_square`, `edge_square`, `corner_square`,
  `light_square`, `dark_square`), king-zone unary truths
  (`u_king_zone(x)`, `t_king_zone(x)`), pseudo-legal-attack
  derivatives (`attacked_by_u_any`, `attacked_by_t_any`,
  `occupied_and_attacked_by_*`), and material-value indicators
  (`u_valuable`, `t_valuable`).
- **18 binary relations**: static geometry
  (`same_rank`, `same_file`, `same_diag`, `knight_step`,
  `king_step`, `rays_align`), occupancy-aware ray relations
  (`between_occupied_count_0`, `between_occupied_count_1`),
  per-side pseudo-legal attack relations
  (`u_attacks`, `t_attacks`, `u_ray_attacks`, `t_ray_attacks`,
  `u_knight_attacks`, `t_knight_attacks`, `u_pawn_attacks`,
  `t_pawn_attacks`), and the king-zone proximity relations
  (`near_t_king`, `near_u_king`).

Sliding-piece relations are gated by between-square occupancy so a
pinned ray collapses to its blocker; pawn double-step and king-zone
relations are similarly clearance-aware.

## Predicate bank

Raw facts are projected through a softmax-mixed predicate bank into

```text
C(x) in [0, 1]^{24}, R(x, y) in [0, 1]^{16}.
```

Each latent concept `C_m` is a convex combination of the 32 raw
unaries and each latent role `R_n` is a convex combination of the 18
raw relations. The mixture is initialised near a one-hot and is free
to drift; mixture entropy is exposed as a diagnostic. This keeps the
predicate set typed and bounded.

## Formula library

A *bounded board hinge formula* is a typed, shallow word in three
families:

```text
F1: exists_x. C_m(x)
F2: exists_x exists_y. C_a(x) AND R_n(x, y) AND C_b(y)
F4: exists_x exists_y. C_a(x) AND R_n(x, y) AND (C_b(y) AND t_king_zone(y))
```

The library is fixed at construction with `(N1, N2, N4) = (24, 96, 48)`
formula instances, giving `F = 168` formulas. The body conjunction is
evaluated with the Lukasiewicz t-norm
`A_AND_B = max(0, A + B - 1)`; the `exists` quantifier is the bounded
soft maximum

```text
exists_x. f(x)  ~  sum_x  softmax(tau * f(.))[x] * f(x), tau > 0,
```

so each formula truth `phi_w(B) in [0, 1]` is a smooth, bounded
function of the board fact tensors and is differentiable in the bank
parameters and the `exists` temperature. Because both `C_m` and `R_n`
are convex combinations and the connective is a Lukasiewicz t-norm,
every truth value stays inside `[0, 1]` *without* the unbounded
saturation drift that motivates the packet's bounded-hinge proposal.

## PSL hinge energy and decision

For a hinge power `p in {1, 2}` define the puzzle decision via a
probabilistic soft-logic energy gap

```text
E(B, y) = bias_y + sum_{w} w_pos(w) * phi_w(B)^p   if y = 0
       = bias_y + sum_{w} w_neg(w) * phi_w(B)^p    if y = 1
```

with `w_pos, w_neg = softplus(...)` non-negative and a learnable
temperature `tau > 0`. The puzzle logit is the energy gap

```text
logit(B) = bias + tau * (E(B, 0) - E(B, 1))
        = bias + tau * sum_{w} ( w_pos(w) - w_neg(w) ) * phi_w(B)^p,
```

so the BCE-with-logits trainer optimises a margin between the
puzzle-positive and puzzle-negative PSL energies. The hinged truth
`phi_w(B)^p` is the bounded surrogate of the PSL hinge potential. The
sign of `w_pos - w_neg` exposes which formulas vote *for* and *against*
the puzzle class; their absolute values rank rule importance.

## Puzzle decision

The puzzle logit is therefore a *hinge function of the formula truths*
only — there is no convolutional trunk, no separate embedding mixer,
no learned MLP head. The trainable parameters are the predicate-bank
concept/role mixtures, the `exists` temperature, the head's per-rule
positive and negative weights, the head bias, and the head temperature.
This produces one BCE logit per board, faithful to the puzzle_binary
target task. The formula truths, the per-rule weights, the energy
gap, and the predicate mixture entropies are exposed as diagnostics
for ablation and interpretability runs.

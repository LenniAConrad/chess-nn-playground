# Math Thesis

Pin / X-Ray / Overload Sheaf -- i252.

Source packet:
`ideas/research/packets/classic/i252_pin_xray_overload_sheaf.md`.

Working thesis: i018's typed cellular sheaf already proves that real
chess relation topology does real work on `puzzle_binary`. The packet's
weakest slices are `pin`, `skewer`, `overload`, and `discovered_attack`
-- exactly the slices i018's 12-plane graph cannot represent as
*conditional* dependencies. i252 keeps the sheaf operator class intact
and adds ten bounded pairwise planes that encode these dependencies
explicitly.

## Setting

Let `N = 64`. Let `o in {0,1}^N` be occupancy, `e = 1 - o`. Let `u_i`
and `t_i` be mover-oriented us / them piece indicators on square `i`.
Let `mu_i` be a fixed piece-order score (pawn `1`, knight `3`, bishop
`3`, rook `5`, queen `9`, king `100`) and `nu_i = min(mu_i, 9) / 9 in
[0,1]` the normalized non-search tactical importance used for overload
weighting.

Let `B_{ijq}` be the existing between-square tensor that i018 uses for
slider visibility, and let `V^{rook}`, `V^{bish}` be i018's visible-ray
masks:

```text
clear_{ij}  = 1[sum_q B_{ijq} * o_q = 0]
V^{rook}_{ij} = R^{rook}_{ij} * clear_{ij}
V^{bish}_{ij} = R^{bish}_{ij} * clear_{ij}
```

The 12 base relations are unchanged from i018.

## Absolute pin, side-specific

i018 builds a symmetric pin bank with templates `m in T_pin` storing
`(slider s_m, blocker d_m, king k_m, line l_m, clear-mask Q^{pin}_m)`.
i018 also emits a single symmetric pin plane `pin_mask`. i252 keeps
that base plane and additionally derives side-specific pins:

```text
Gamma^{pin}_m   = 1[sum_q Q^{pin}_{mq} * o_q = 0]
P^{us}_{sd}     = clip( sum_m 1[s=s_m] 1[d=d_m]
                        Gamma^{pin}_m
                        S^{us}_{s,l_m}
                        t_d
                        K^{them}_{k_m},  0, 1 )
P^{them}_{sd}   = clip( sum_m 1[s=s_m] 1[d=d_m]
                        Gamma^{pin}_m
                        S^{them}_{s,l_m}
                        u_d
                        K^{us}_{k_m},   0, 1 )
```

Here `S^{us}_{s, rook}` means "our rook-or-queen on `s`" and
`S^{us}_{s, bish}` means "our bishop-or-queen on `s`". `K^{us}_k` is the
us-king indicator on `k`, mirror for `K^{them}`. These two side-specific
masks are used to derive the new dependency planes below.

## Single-screen template bank

The single-screen template bank `T_1` contains every ordered triple
`m = (s_m, c_m, r_m, l_m)` on an aligned rook or bishop ray with the
screen strictly between source and rear. The clear mask `Q^1_m`
contains every square strictly between `s_m` and `r_m` *except* `c_m`.
On an 8x8 board this gives exactly `2576` templates. Per batch:

```text
Gamma_m = 1[sum_q Q^1_{mq} * o_q = 0]
```

## New dependency planes

X-ray, skewer, discovered:

```text
X^{us}_{sr} = clip( sum_m 1[s=s_m] 1[r=r_m]
                     Gamma_m
                     S^{us}_{s,l_m}
                     o_{c_m}
                     t_r,                          0, 1 )

K^{us}_{sr} = clip( sum_m 1[s=s_m] 1[r=r_m]
                     Gamma_m
                     S^{us}_{s,l_m}
                     t_{c_m}
                     t_r
                     1[mu_{c_m} > mu_r],           0, 1 )

D^{us}_{cr} = clip( sum_m 1[c=c_m] 1[r=r_m]
                     Gamma_m
                     S^{us}_{s_m,l_m}
                     U^{us,nonking}_c
                     t_r
                     (1 - pin^{us}_c),             0, 1 )
```

with `pin^{us}_c = clip(sum_s P^{them}_{sc}, 0, 1)` (our piece on `c`
absolutely pinned by their slider).

Pinned-defender exposure and overload exposure:

```text
pin^{them}_d  = clip( sum_s P^{us}_{sd}, 0, 1 )

rho^{us,pdef}_r = clip( sum_d Def^{them,nonking}_{dr} * pin^{them}_d, 0, 1 )

crit^{us}_r        = nu_r * 1[(sum_s A^{us}_{sr}) > 0]
g^{them}_{dr}      = Def^{them,nonking}_{dr} * crit^{us}_r
omega^{them}_d     = second_largest_r g^{them}_{dr}
rho^{us,ovl}_r     = clip( sum_d Def^{them,nonking}_{dr} * omega^{them}_d, 0, 1 )

M^{us,pdef}_{sr} = A^{us}_{sr} * them_r * rho^{us,pdef}_r
M^{us,ovl}_{sr}  = A^{us}_{sr} * them_r * rho^{us,ovl}_r
```

(All `them_*` mirrors swap us / them throughout.) `Def^{them,nonking}`
is `them_attack * them_piece * them_nonking` (defenders that are not the
king); this matches the packet's intent that the overload term measures
*nontrivial* second duty.

The five `us`/`them` pairs above give the 10 new typed planes.

## Sheaf operator

The diffusion block is the i249 algebraic block parameterised by the
22-element `RELATION_SIGNS_V2` and the relation count `22`. Sign
assignment keeps `+1` only for the same-side defense planes
(`us_defends_us_piece`, `them_defends_them_piece`); every other plane,
including the ten new ones, gets `-1`, consistent with i018's policy
that attack-flavored planes get the negative sign.

## Hypothesis

If i018's typed topology is right and the slice-specific gap is the
absence of conditional tactical dependencies, then adding these ten
bounded planes to the same sheaf math should:

- not regress overall PR-AUC by more than `0.003` on `puzzle_binary`,
- improve at least one of: matched-recall near-puzzle FP at recall
  `0.80` or `0.85`, or mean PR-AUC across the four target motif slices
  (`pin`, `skewer`, `overload`, `discovered_attack`) by `>= 0.010`,
- show the gain disappear under the dependency-only scramble
  (`scramble_new_only: true`).

## Falsifiers

| ID | Switch | What it tests |
|---|---|---|
| F1 | i018 / i249 baseline | Reference. |
| F2 | `model.scramble_relations: true` | i018's typed-topology falsifier. |
| F3 | `model.scramble_new_only: true` | Only the new dependency planes are scrambled. If the lift survives, the new family is decorative. |
| F4 | `model.family_collapse: true` | All 10 new planes collapse to a generic dependency average. If matched, typing is unnecessary. |

The packet's `value-blind skewer / overload`, `no overload planes`,
`no pinned-defender planes`, and `no self-pin legality filter` ablations
are documented in `ablations.md` and require a forward edit (not config).

## Decision rule

Treat i252 as a meaningful improvement over i249 / i018 only if all
three hold across seeds 42, 43, 44 at base scale, with all other
hyperparameters matched:

- overall PR AUC is not worse than i249-fast by more than `0.003`;
- matched-recall near-puzzle FP improves at recall `0.80` or `0.85`,
  OR mean PR AUC across `pin`, `skewer`, `overload`,
  `discovered_attack` slices rises by `>= 0.010`;
- F3 (dependency-only scramble) loses most of the lift.

If those conditions are not met, do not retrain longer -- the richer
graph did not buy enough signal. This is the discipline the packet
asked for.

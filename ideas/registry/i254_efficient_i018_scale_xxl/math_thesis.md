# Math Thesis

i254 keeps the i018 oriented-tactical-sheaf object exactly and asks one
controlled question about scale: does i018 still have profitable
headroom when the extra parameter budget goes into width (regular dense
work) instead of stalk size or sheaf depth (irregular relation work)?

## Inherited Object

The cell complex, stalks, sheaf restriction maps, signs, gates, heat
step, triad-defect pool, and readout are inherited from i018:

- 64 square 0-cells;
- 12 typed tactical relations `M_r` (attacker/defender, king-zone,
  slider rays, knight, oriented pawn, pin candidate);
- per-relation source/target restriction matrices `rho_src[r]`,
  `rho_dst[r]`;
- fixed signs `sigma_r in {-1, +1}`;
- bounded gates `g_r = 2 * sigmoid(logit_r)` and heat step
  `eta = 0.25 * sigmoid(eta_logit)`;
- the same triad-defect pool and readout diagnostics.

The i018 sheaf block uses the standard relation-weighted coboundary

```text
(delta_rho h)_{(u,v,r)} = sqrt(w_uvr) * (rho_dst_r h_v - sigma_r rho_src_r h_u)
```

and the sheaf Laplacian step is `delta_rho^T delta_rho`. The Laplacian
is symmetric positive semidefinite for any choice of linear
restrictions, so the grouped low-rank parameterization considered below
does **not** break the sheaf-thesis itself.

## Two Restriction Modes

Let `s = stalk_dim`, `R = 12`, and `r` index relations.

`restriction_mode="full"` (default first XXL run):

```text
rho_src[r], rho_dst[r] in R^(s x s),  initialised at I + 0.02 * noise
```

This is the i018 parameterization, unchanged. State-dict layout
matches i018 exactly so i018 checkpoints load into i254 in `full` mode.

`restriction_mode="grouped_lowrank"` (optional, behind a config flag):

Partition the 12 relations into G groups by semantic role. The default
partition is G=4:

```text
attack:  us_attacks_them_piece, them_attacks_us_piece,
         us_attacks_empty_near_king, them_attacks_empty_near_king,
         knight_attack, pawn_attack_forward_oriented      (6 relations)
defense: us_defends_us_piece, them_defends_them_piece     (2 relations)
ray:     bishop_ray_visible, rook_ray_visible, queen_ray_visible (3)
pin:     king_ray_pin_candidate                            (1 relation)
```

For each group g, share `U_src_g, V_src_g, U_dst_g, V_dst_g in R^(s x k)`.
For each relation r in group g(r), use a relation-specific diagonal
`a_src_r, a_dst_r in R^k`. The restriction maps are

```text
rho_src[r] = I_s + U_src_{g(r)} diag(a_src_r) V_src_{g(r)}^T
rho_dst[r] = I_s + U_dst_{g(r)} diag(a_dst_r) V_dst_{g(r)}^T
```

The materialised `(R, s, s)` tensor is recomputed each forward and the
rest of the block reduces to the same matrix products as the full
case. This is parameter sharing, not a fused custom kernel.

## Parameter Arithmetic

Restriction-map parameters per block:

| Mode | Formula | Value at `s=8`, `k=4`, `G=4` |
|---|---|---:|
| `full` | `2 * R * s^2` | `2 * 12 * 64` = **1,536** |
| `grouped_lowrank` | `4 * G * s * k + 2 * R * k` | `4 * 4 * 8 * 4 + 2 * 12 * 4` = **608** |

That is a ~60% map-parameter reduction in grouped mode. At the current
`s=8` the static compute benefit of low-rank application is small (the
dominant `64 * 64 * s` edge work remains), so the grouped low-rank
family is mainly useful as a *safety mechanism for a future stalk
scaling*: it prevents restriction-map cost from growing quadratically
with `s`.

## What Scales and Why

| Knob | First XXL action | Rationale |
|---|---|---|
| `channels` | 128 -> 160 | regular dense capacity; historically helped i018 |
| `hidden_dim` | 192 -> 320 | readout dominates dense params; cheapest extra capacity |
| `depth` | 4 -> 4 (unchanged) | every extra block multiplies relation work |
| `stalk_dim` | 8 -> 8 (unchanged) | stalk scaling multiplies `64 * 64 * s` work |
| `restriction_mode` | `full` (unchanged) | first XXL has no parameterization confound |

This is the synthesis of the repo's architecture, the falsifier, the
i249 postmortem, and the static parameter arithmetic.

## Memory Geometry

The fixed relation tensor has shape `(B, 12, 64, 64)`. At batch size
128 it is about 12 MiB in bf16. The eager sheaf block processes one
relation at a time, so a single `(B, 64, 64, s)` residual at `s=8` is
about 8 MiB in bf16 at batch 128.

Width scaling does **not** change those `12 * 64 * 64` relation
tensors at all. Stalk scaling does. That is the second reason the
first XXL run leaves `stalk_dim` alone.

## Equivalence Claim with i018

When `restriction_mode = "full"` and the trunk hyperparameters match
i018 exactly, the forward computation of i254 is **bit-identical** to
i018: the state_dict shape, parameter init, signs, gates, heat step,
and per-relation loop are unchanged. This is verified in the test
suite by loading i018 weights into i254 (`strict=True`) and asserting
zero logit / sheaf_tension diff on a fixed input.

## Falsifiers

- **Scale falsifier.** If the 3-seed i254 capacity run does not beat
  i018 scale_xl's 0.8901 mean PR-AUC by at least +0.003, the family
  does not have profitable scale runway in this direction.
- **Stalk falsifier.** If any `s > 8` variant (full or grouped) fails
  to beat the best `s=8` width/head-scaled variant, stalk scaling is
  unsupported for this family.
- **Grouped-map falsifier.** If grouped low-rank restrictions
  underperform same-budget full maps by more than seed noise, drop
  them from the mainline.
- **Systems falsifier.** If profiling shows the incidence builder is
  not a top hotspot, do not write a fused incidence kernel. If
  compile-only gives negligible benefit, do not attribute later speed
  changes to compile.
- **Parity falsifier.** Any execution-only rewrite (compile, fusion,
  custom kernel) must pass the train-mode mixed-precision parity
  ladder before being benchmarked for speed. The ladder is described
  in the source research markdown.

## Failure Modes That Drop i254

- The 3-seed capacity run fails the scale falsifier. The family has
  plateaued and the next move is a structural change, not a wider
  trunk.
- The grouped low-rank variant beats the full variant only in
  conjunction with stalk scaling, while pure stalk-only diagnostics
  fail. That means the gain is coming from the parameter sharing, not
  the stalk size, and the simplest interpretation is that the family
  was over-parameterised at the original `s=8`.
- Compile-only A/B on the unmodified baseline gives a meaningful
  speedup. That implies the bottleneck was Python overhead, not the
  algebra, and a fused incidence kernel will not help; the next move
  is `torch.compile`, not custom code.

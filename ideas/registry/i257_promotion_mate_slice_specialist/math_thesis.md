# Math Thesis

`Promotion Mate Slice Specialist` -- i257.

The benchmark contract is binary puzzle classification with the source-1
class held out as verified near-puzzle. Let `y in {0, 1}` denote the binary
label and let `s` denote the per-board score. The deployment metric set
includes overall PR-AUC, slice PR-AUC on `promotion`, `underpromotion`, and
`mate_in_1`, and matched-recall near-puzzle false-positive rate at
`puzzle_recall in {0.80, 0.85}`.

## Decomposition Identity

Let the shared trunk map board `x` to a base logit `z0(x)`. Let the
specialist set be `K = {prom, under, mate, joint}`. Each branch outputs

```text
delta_k(x) = Delta_k * tanh( v_k^T s_k(x) )           bounded in [-Delta_k, Delta_k]
gate_k(x)  = m_k(x) * sigmoid( a_k^T [s_k(x), c_k(x)] + b_k )   in [0, 1]
z(x)       = z0(x) + sum_k gate_k(x) * delta_k(x)
```

with `m_k(x) in {0, 1}` a structural mask that prevents impossible
specialists from firing, `s_k(x)` the branch summary, and `c_k(x)` a
branch-confidence scalar. The `tanh` and `Delta_k` bound together cap each
branch's contribution, and `sigmoid` keeps gates non-negative and bounded.
This is the central identity: no specialist can grow without bound, and
each one only fires when its prerequisites hold.

## Promotion Candidate Field

The candidate set `C(x)` for promotion is built deterministically from the
simple_18 board. simple_18 stores white pawns on plane 0 and black pawns on
plane 6 with array row index 0 corresponding to chess rank 8. For the side
to move:

- white pawns one push from promotion are on row 1, two pushes on row 2;
- black pawns one push from promotion are on row 6, two pushes on row 5.

The candidate gather takes the top-K (default `K = 4`) positive entries
from the relevant 16-square slab (one-push slab concatenated with two-push
slab) selected by the side-to-move plane.

For each candidate `j` we form a candidate descriptor

```text
descriptor_j = MLP_proj([trunk[source_j], exchange[source_j], king[source_j],
                         rank_distance_j, own_attacks(promo_sq_j),
                         enemy_attacks(promo_sq_j),
                         enemy_zone(promo_sq_j),
                         own_ray_to_zone(promo_sq_j),
                         capture_promo_indicator_j,
                         src_attacker_pressure_j,
                         src_defender_pressure_j])
```

The promotion-square statistics are gathered from the deterministic
exchange and king feature stacks using the destination-square index that
matches each candidate.

## Promotion Fanout (Q / R / B / N)

For each candidate `j` and type `t in {Q, R, B, N}` we form a
type-conditioned descriptor

```text
u_{j,t} = MLP_type([descriptor_j, type_emb(t), per_type_attack_delta_{j,t}])
```

where `per_type_attack_delta_{j,t}` masks the 6-dimensional analytic
attack-delta vector by a per-type weight (queen takes all components, rook
discounts diagonals, bishop discounts rank-file, knight zeroes ray
components). A per-(candidate, type) score `s_{j,t} = w^T u_{j,t}` feeds
softmax type attention over `t`, and the type-weighted descriptors are
pooled across candidates with the candidate mask:

```text
alpha_{j,t}     = softmax_t(s_{j,t})
weighted_j      = sum_t alpha_{j,t} * u_{j,t}
pooled_desc(x)  = sum_j w_j * weighted_j / sum_j w_j           (mask-weighted)
```

The promotion summary head consumes `pooled_desc(x)` plus six pool
statistics (`candidate_count`, aggregate type-score max / mean,
type-attention entropy, best-type peak attention, mask-norm clamp) and
emits `delta_prom` (bounded) and a gate input that consumes the summary
plus a confidence scalar `agg_type_max`.

## Underpromotion Divergence

The same per-(candidate, type) scores feed a non-queen-vs-queen margin:

```text
queen_score_j    = s_{j, Q}
nonqueen_score_j = max(s_{j, R}, s_{j, B}, s_{j, N})
margin_j         = nonqueen_score_j - queen_score_j
agg_margin       = sum_j w_j * margin_j / sum_j w_j
```

The summary head consumes `[agg_margin, candidate_count, type_entropy,
agg_nonqueen, agg_queen]`. The bounded delta and sigmoid gate share the
same shape as the promotion branch. If `agg_margin` is strongly negative
the branch learns to stay silent; if it is positive in a position with
candidates the branch can add a nontrivial `delta_under`. This is the
direct architectural translation of the promotion packet's central
insight: the supervision problem is not "is there a pawn near promotion?"
but "which future piece identity changes the tactical geometry?"

## King-Zone Forcing Witness

The witness branch pools the trunk activations weighted by the deterministic
enemy-king mask and enemy-king-zone mask:

```text
trunk_at_enemy_king = sum_s enemy_king(s) * trunk(s) / count(enemy_king)
trunk_at_enemy_zone = sum_s enemy_zone(s) * trunk(s) / count(enemy_zone)
```

and concatenates them with the deterministic king feature mean and the
dual-stream board-level summary, plus six explicit witness scalars derived
from the deterministic features:

```text
check_count     ~ number of own attacks on the enemy king square
escape_count    ~ enemy king-zone squares not under own attack and empty
in_zone_attack  ~ own attacking pressure mass inside the enemy zone
own_ray_count   ~ own ray squares pointing into the enemy zone
capture_value   ~ enemy material in the zone vulnerable to own attacks
zone_balance    ~ enemy zone occupancy ratio (clamped to 1.0)
```

A small MLP emits the bounded delta and a sigmoid gate, conditioned on
`confidence = (check_count + in_zone_attack) / 16`. The structural mask
zeros the branch unless at least one of `check_count` or `in_zone_attack`
is positive.

## Promotion-Mate Joint Branch

```text
joint_input = [promo_summary, under_summary, mate_summary]
joint_summary = MLP_joint(joint_input)
delta_joint   = Delta_joint * tanh( v_joint^T joint_summary )
gate_joint    = sigmoid( a_joint^T [joint_summary, c_joint] + b_joint )
                * structural_mask
```

The structural mask requires both the promotion and the mate branch to be
non-trivial (candidate count > 0 and witness count > 0). The confidence
scalar is the product of the promotion and mate gates (detached so
back-propagation does not double-count the upstream gate parameters).

## Loss Identity

For the current shared-trainer implementation the loss is just

```text
L = BCEWithLogits(final_logit, y_binary)
```

The research packet's extended loss
`L = L_BCE + lambda_gate * sum_k E[gate_k] + lambda_kd * T^2 KL(...) +
lambda_near * L_near + lambda_slice * L_slice` requires a trainer extension
to inject per-batch pair-matched terms. Those terms are not mathematically
required to make `final_logit` well-defined, and adding them would couple
the architecture promotion to a trainer change the rest of the queue does
not need. The keep / drop rule for that extension is documented in
`ablations.md`.

## Falsifiers

- `trunk_only` ties or beats `none` on slice PR-AUC: the specialist is not
  load-bearing and should be dropped.
- `copy_baseline_fanout` ties `none` on `promotion` / `underpromotion`
  slices: type-conditioned fanout is not load-bearing -- drop the
  promotion branch.
- `uniform_type_attention` ties `none`: selective type weighting is
  decorative -- simplify the branch.
- `zero_under_margin` ties `none`: the non-queen-vs-queen margin is not
  load-bearing -- drop the underpromotion branch.
- `no_mate_witness` ties `none` on `mate_in_1`: the king-zone witness
  scalars are not load-bearing -- drop the mate branch.
- `no_joint_branch` ties `none`: the joint overlap branch is decorative --
  drop it.
- `disable_gate` worsens near-puzzle FP rate at matched recall: the gate
  is structurally important.
- `force_zero_gate` recovers the trunk baseline closely: the wrapper has
  no leakage into the base logit.
- Final logit fails the bounded-delta identity (`|final - base| > sum_k
  Delta_k` for some sample): a refactor has broken the bounding contract.

If any falsifier trips, the responsible branch is reduced or removed. The
keep condition is matched-recall slice lift on `promotion`,
`underpromotion`, and `mate_in_1` versus the matched i193 parent on the
canonical tagged split, evaluated with thresholds chosen on validation.

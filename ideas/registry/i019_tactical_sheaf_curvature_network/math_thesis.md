# Math Thesis

Tactical Sheaf Curvature Network (`TSCN`)

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0254_tuesday_local_tactical_sheaf_curvature.md`.

## Working thesis

Puzzle-likeness in the `puzzle_binary` task is detectable as localized
inconsistency on a typed directed candidate-relation complex over the 64
board squares. The bespoke implementation realizes that thesis as a learned
sheaf coboundary on chess-geometric relations, a Laplacian-like bounded
diffusion step, and a target-centered curvature readout.

## Object

Let an input position be `X in R^(C x 8 x 8)`. After flattening into the
square set `V = {0, ..., 63}` a learned adapter produces node states
`h_v in R^{d_node}` and a stalk projection produces `x_v = P h_v in R^s`.

A fixed directed typed relation set `E` is built from chess geometry
alone (no engine, no labels). It contains:

- Sliding-ray candidates along ranks, files, and diagonals at distances
  `1, 2, 3, 4+`.
- Knight-displacement candidates.
- King-neighborhood candidates.
- Oriented pawn-capture candidates kept as separate relation types so
  pawn direction is preserved.

Every directed edge `e = (u -> v, r)` carries an integer `edge_type`, a
`relation_group` ID for group-pooled statistics, and a normalized
geometry vector (source / target ranks and files, signed deltas, absolute
distance, distance bucket).

## Sheaf operator

Per layer `l`, a learned `SheafRestrictionGenerator` reads the relation
geometry and emits diagonal restrictions

```text
rho_src(e) = diag(a_l(q_r))
rho_dst(e) = diag(b_l(q_r))
```

bounded as `1 + 0.5 * tanh(.)` so the coboundary stays well-conditioned at
initialization. An input-dependent gate

```text
g_e = sigmoid(Gate_l(x_u, x_v, q_r))
```

selects which candidate relations participate. The (signed) coboundary is

```text
(delta_l x)_e = rho_dst(e) x_v - rho_src(e) x_u,
```

and the weighted sheaf energy is

```text
E_l(x) = sum_{e in E} g_e * ||(delta_l x)_e||_2^2.
```

The diffusion update is the bounded residual step

```text
x <- LayerNorm(x - eta_l * D_l^{-1} delta_l^T G_l delta_l x + NodeMLP_l(x)),
```

with `G_l = diag(g_e)`, `D_l` a per-node degree normalizer, and `eta_l`
clamped through a sigmoid so the heat step is non-expansive in the
quadratic-energy sense.

## Target-centered curvature

For every destination square `v` the layer also computes the gate-weighted
variance of the transported source claims:

```text
z_e = a * x_u for incoming edges e = (u -> v),
curv_v = weighted_variance({z_e : head(e) = v}).
```

This is a soft all-pairs disagreement among incoming tactical claims on the
same target and acts as a cheap 2-cell / higher-order interaction proxy.

## Hypothesis

Puzzle-like positions concentrate sheaf frustration `E_l` and target-centered
curvature on typed chess-geometric relations after shallow learned diffusion,
while non-puzzles tend to have more mutually compatible local constraints.
The classifier consumes mean / max / std node pools plus per-layer and
per-relation-group statistics of edge energy, gate strength, gate entropy,
and curvature, returning one BCE puzzle logit plus reporting-only
diagnostics.

## What is proven

- For fixed gates and bounded restrictions, the sheaf Laplacian
  `L = delta^T G delta` is positive semidefinite and the bounded heat step
  `x <- x - eta L x` is non-expansive in quadratic energy for sufficiently
  small `eta`.
- Tying parameters across the file mirror gives equivariance under the
  corresponding relation-complex automorphism (left-right reflection of the
  rank axis applied consistently to the tied relation classes).

## What is hypothesized

- High learned sheaf frustration and target-centered curvature correlate
  with puzzle-likeness on the CRTK benchmark distribution.
- Learned gates suppress irrelevant geometric candidate edges and emphasize
  chess-useful relations without engine analysis.

## Falsifiers and counterexamples

- Quiet endgame studies, zugzwang positions, fortress breaks, and long
  strategic puzzles can be puzzle-like with weak immediate line tension.
- Tactical melees may exhibit high attack/defense curvature without being
  verified puzzles.
- The central falsifier (per the source packet) is replacing the typed
  sheaf coboundary with parameter-matched typed gated message passing on
  the same relation list. If the falsifier matches TSCN within `0.005`
  ROC-AUC and macro-F1 on the same seeds, the sheaf-frustration thesis is
  rejected.

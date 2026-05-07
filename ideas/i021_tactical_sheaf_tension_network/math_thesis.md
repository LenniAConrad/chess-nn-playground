# Math Thesis

Tactical Sheaf Tension Network (`TSTN`)

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0255_tuesday_local_tactical_sheaf.md`.

## Working thesis

Puzzle-likeness in the `puzzle_binary` task is detectable as the residual
tension of a side-aware cellular sheaf placed over pseudo-legal attack,
defense, control, and one-blocker x-ray relations on the current board. The
bespoke implementation realizes that thesis as a typed directed tactical
complex with diagonal-plus-low-rank relation-tied restrictions, a bounded
sheaf-Laplacian heat step, and a Dirichlet-energy readout consumed by the
classifier head.

## Object

Let an input position be `X in R^(C x 8 x 8)`. The square set is
`V = {0, ..., 63}`. A learned `SquareStalkEncoder` produces stalk states
`x_v in R^{fiber_dim}` from the per-square encoding, the side-to-move flag,
the per-square piece-type one-hot, the per-square role one-hot, and the
deterministic 6-D square-coordinate features.

A directed typed relation set `E` is built from chess geometry alone (no
engine, no labels). It contains, per source piece:

- Pawn forward-diagonal capture candidates oriented by side-to-move.
- Knight offsets and king-neighborhood offsets.
- Slider rays (bishop / rook / queen) walked until the first blocker;
  that blocker generates a control / attack / defend edge depending on
  target color, and a one-blocker x-ray edge is emitted past the blocker
  when a second occupied square is reachable along the same ray.

Each directed edge `e = (u -> v, tau)` carries an integer relation type
`tau in [0, RELATION_COUNT)` indexed by
`(role, piece, edge_kind, direction_bin)` and a relation group used for
group-pooled tension statistics. Direction bins fold east / west and
NE / NW (and SE / SW) into shared bins by default so the only tied symmetry
is left-right file reflection; pawn direction, castling, and side-to-move
asymmetries are not tied. Each edge also carries a degree-normalized
weight `w_e = (deg(u) * deg(v))^{-1/2}`.

## Sheaf operator

For each relation `tau`, the diagonal-plus-low-rank restriction maps are

```text
R_src(tau) = diag(d_src[tau]) + U_src[tau] V_src[tau]^T
R_dst(tau) = diag(d_dst[tau]) + U_dst[tau] V_dst[tau]^T
```

with rank `restriction_rank` typically much smaller than `fiber_dim`. The
(signed) coboundary on `e = (u -> v, tau)` is

```text
(delta x)_e = R_src(tau) x_u - R_dst(tau) x_v,
```

and the weighted sheaf Dirichlet energy is

```text
E(x) = sum_{e in E} w_e * ||(delta x)_e||_2^2.
```

The diffusion update is the bounded residual step

```text
x <- LayerNorm(x - eta * D^{-1} delta^T W delta x + NodeMLP(x)),
```

with `W = diag(w_e)`, `D` a per-node edge-weight degree normalizer, and
`eta` clamped through a sigmoid so the heat step is non-expansive in the
quadratic-energy sense. `delta^T` is realized concretely by
`R_src^T(tau) (w_e * delta_e)` scattered to source squares minus
`R_dst^T(tau) (w_e * delta_e)` scattered to destination squares.

## Side-aware tension readout

Per block the model exposes mean / weighted-mean / max / top-3 edge tension,
the normalized edge count, and per-relation-group mean tension over
`(control_empty, attack_enemy, defend_own, xray_one_blocker, king_ring)`.
Edges that target the king-ring of either side are promoted to the
`king_ring` group so king-pressure tension is separable. The pool also
returns side-to-move-only and opponent-only weighted node pools so the
classifier sees a side-aware summary of the residual tension surface.

## Hypothesis

Puzzle-like positions concentrate sheaf-coboundary tension on typed
chess-geometric relations after shallow learned diffusion, while
non-puzzles tend to have more mutually compatible local tactical claims.
The classifier consumes the side-aware node pools plus the per-block and
per-relation-group tension statistics, returning one BCE puzzle logit plus
reporting-only diagnostics.

## What is proven

- For fixed restrictions and weights, `L = delta^T W delta` is positive
  semidefinite and the bounded heat step `x <- x - eta L x` is non-expansive
  in quadratic energy for sufficiently small `eta`; the sigmoid-bounded
  `eta` keeps the per-block update stable.
- Tying parameters across the file-mirror direction bin gives equivariance
  under the corresponding relation-complex automorphism (left-right
  reflection of the file axis applied consistently to the tied direction
  classes).

## What is hypothesized

- High learned sheaf tension on typed chess-geometric edges correlates with
  puzzle-likeness on the CRTK benchmark distribution.
- Group-pooled tension on king-ring and attack edges adds usable signal
  beyond the mean / max / std node pool of a generic CNN trunk.

## Falsifiers and counterexamples

- Quiet endgame studies, zugzwang positions, fortress breaks, and long
  strategic puzzles can be puzzle-like with weak immediate line tension.
- Tactical melees may exhibit high attack/defense tension without being
  verified puzzles.
- The central falsifier from the source packet is replacing the typed
  sheaf coboundary with parameter-matched typed gated message passing on
  the same relation list. If the falsifier matches `TSTN` within `0.005`
  ROC-AUC and macro-F1 on the same seeds, the sheaf-tension thesis is
  rejected.

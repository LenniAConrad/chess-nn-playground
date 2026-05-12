# Math Thesis

Entropic Chess Geometry Transport Network (ECGT-Net).

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0703_tuesday_los_angeles_geom_ot.md`.

## Thesis

Puzzle-like positions often contain an unusually organized transport
geometry from the side-to-move material toward opponent tactical target
zones. ECGT-Net tests this hypothesis with a deterministic
chess-distance cost matrix and a small entropic Sinkhorn transport
layer over the 64 current-board squares, fused with a compact CNN
trunk. The operator never reads engine evaluations, search lines,
legal-move counts, one-ply move-delta bags, verification metadata, or
dataset provenance.

## Setup

For a board `x` (encoding `simple_18`) we extract side-to-move
canonical piece planes and decompose them into

- side-to-move *source atoms*
  `S(x) = { (p_i, s_i) : i = 1..m(x) }` where `p_i` is a piece type
  in `{pawn, knight, bishop, rook, queen, king}` and `s_i` is a
  square in `{1..64}`,
- opponent *target atoms*
  `T(x) = { (r_j, t_j) : j = 1..n(x) }` with target role `r_j` in
  `{king_square, king_ring, heavy_piece, minor_piece, pawn,
  promotion_anchor}` and square `t_j`.

Source and target marginals `a(x)` and `b(x)` are formed from
nonnegative learned type/role weights, masked by atom existence, and
normalized to the simplex. No label, engine score, or source metadata
enters these marginals.

A board-conditioned cost is

```text
C_x(i,j) = softplus(alpha_{p_i, r_j}) * d_{p_i}(s_i, t_j)
         + beta_{p_i, r_j}
         + softplus(gamma_{p_i, r_j}) * delta_{Manhattan}(s_i, t_j),
```

where `d_p` is a deterministic empty-board chess-distance table for
piece type `p` (knight BFS, rook line distance, bishop colour-aware
distance, queen min(rook, bishop), king Chebyshev, directional pawn
distance with cap), `alpha_{p,r}` and `gamma_{p,r}` are non-negative
softplus parameters, and `beta_{p,r}` is a learned offset. Invalid
source/target pairs receive a large masked cost.

The entropic transport plan solves

```text
Pi_eps(x) = argmin_{pi >= 0}  <pi, C_x>
            + eps * sum_{i,j} pi_{ij} (log pi_{ij} - 1)
```

subject to `pi @ 1 = b(x)` and `pi.T @ 1 = a(x)`. The implementation
unrolls a fixed log-domain Sinkhorn loop with `eps > 0`.

From `Pi_eps(x)` ECGT-Net builds

- the type-role flow matrix
  `M_{p,r}(x) = sum_{i: p_i = p} sum_{j: r_j = r} Pi_{ij}`,
- the scalar features
  `<Pi_eps, C_x>`, normalized plan entropy, max row mass, max column
  mass, valid source count, valid target count,
- a source pressure map by scattering row sums to the source squares,
- target pressure maps by scattering column sums to the target squares
  by role.

These features are fused with the raw board tensor by a small CNN
trunk and a tabular transport MLP before the binary classifier.

## Proposition 1: Equivariance and Invariant Pooled Summaries

Let `g` be a file-mirror or valid side-to-move perspective transform
acting on squares and roles by permutations `P_g` (sources) and `Q_g`
(targets), preserving cost and marginals:

```text
C_{g.x} = P_g C_x Q_g.T,    a(g.x) = P_g a(x),    b(g.x) = Q_g b(x).
```

For `eps > 0` the entropic OT problem is strictly convex on the
positive transport polytope, so the minimizer is unique. If `pi` is
feasible for `(a, b, C_x)` then `P_g pi Q_g.T` is feasible for the
permuted instance, with the same linear cost and entropy. Therefore

```text
Pi_eps(g.x) = P_g Pi_eps(x) Q_g.T.
```

Consequently, type-role pooled flows and scalar costs are invariant
under the transform, while square pressure maps are equivariant.

## Proposition 2: Central Falsification

Let `N(X)` be nuisance counts: source count, target count, material
histogram, target-role histogram, and per-type-role cost histograms.
Let `T(X)` be ECGT-Net's true transport summaries and `T_rand(X)`
their cost-histogram-preserving randomization. If

```text
P(Y | N(X), T(X)) = P(Y | N(X), T_rand(X))
```

almost surely, the transport geometry carries no label-relevant
signal beyond nuisance shortcuts. A measured gap between ECGT-Net and
its cost-randomized ablation is evidence that the chess-distance
geometry is what helps.

## Counterexamples

- Quiet endgame studies whose key feature is zugzwang or opposition
  rather than material moving toward a tactical target.
- Tactics that depend on a single legal constraint that empty-board
  distances cannot see (pinned blockers, stalemate tricks,
  underpromotion, exact castling legality).
- Positions with geometrically promising piece placement where every
  candidate tactic fails for non-geometric reasons.
- Puzzles whose target is not opponent material or king but a
  defensive resource invisible to occupancy-based atom builders.
- Dataset artifacts where positives and negatives have similar target
  geometry but differ by metadata that is correctly forbidden as
  input.

## Self-Critique

The strongest objection is that ECGT-Net could learn a sophisticated
material/proximity heuristic. The cost-histogram-preserving
randomization, uniform-cost OT, count-only, and zero-OT
same-parameter ablations are designed to expose this failure mode.
ECGT-Net is still worth running because it is small, label-safe, and
mathematically distinct from the imported sheaf, attack-graph, and
move-delta families, and because the central falsification ablation
is unusually sharp.

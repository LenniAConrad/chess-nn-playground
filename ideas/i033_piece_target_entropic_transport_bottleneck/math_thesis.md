# Math Thesis

Piece-Target Entropic Transport Bottleneck (PT-ETB).

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0657_tuesday_los_angeles_piece_transport.md`.

## Thesis

A puzzle-like position should expose a low-entropy, asymmetric transport
plan from the side-to-move's existing pieces to opponent pieces (and back
again), even without engine scores, attack/sheaf complexes, or one-ply
move-delta enumeration. PT-ETB tests this hypothesis with a type-aware
entropic optimal-transport bottleneck over the 64 current-board squares
under a side-to-move-relative orientation.

## Setup

Let `S = {1, ..., 64}` index squares in side-to-move-canonical
coordinates, with friendly back rank at row 7 and enemy back rank at row
0. Let `P = {pawn, knight, bishop, rook, queen, king}`. For a board
position `x` and color `c ∈ {friendly, enemy}` write
`o_{c,p,s}(x) ∈ {0,1}` for whether color `c` has a piece of type `p` on
square `s`.

Define type-weighted source/target measures with epsilon-regularized
positivity

```text
mu_x(s) = [eps + sum_p softplus(a_p) o_{friendly,p,s}(x)] / Z_mu(x),
nu_x(t) = [eps + sum_q softplus(b_q) o_{enemy,q,t}(x)]    / Z_nu(x),
```

and define the reverse pair `mu'_x, nu'_x` by swapping friendly and
enemy. The transport cost is

```text
C^h_theta(s,t;x) = softplus(MLP_theta(e_src(s,x), e_tgt(t,x), g(s,t), d^h)) + c_min,
```

where `e_src` and `e_tgt` are square-mixed piece-type embeddings,
`g(s,t)` is deterministic chessboard geometry (file/rank deltas,
Chebyshev/Manhattan distances, same-file/rank/diag indicators, knight
vector, forward relation, source/target centrality), `d^h` is a forward/
reverse direction embedding, and `c_min > 0` is a small floor.

The entropy-regularized transport plan is

```text
pi^h_theta(x) = argmin_{pi in Pi(mu_x, nu_x)} <pi, C^h_theta(.,.;x)>
                + tau * sum_{s,t} pi_{s,t}(log pi_{s,t} - 1).
```

PT-ETB extracts a low-dimensional bottleneck `z(x)` from plan statistics,
projected plan-derived board maps, and the shallow board adapter, and
predicts the binary puzzle label `Y(x)` from `z(x)`. The chess hypothesis
is

```text
I(Y ; transport_geometry(X) | material(X), side_to_move(X), occupancy_marginals(X)) > 0,
```

i.e. piece-target proximity and concentration carry label-relevant signal
beyond material and square-count shortcuts.

## Proposition 1: Well-Posed Differentiable Transport Layer

For `tau > 0`, strictly positive marginals, and bounded finite cost, the
entropic problem has a unique strictly positive minimizer
`pi = diag(u) exp(-C/tau) diag(v)`. Sinkhorn scaling alternately matches
row and column marginals; in log-domain it is numerically stable and
differentiable in `C`, `mu`, and `nu`.

Proof sketch: entropy regularization makes the objective strictly convex
on the transport polytope. KKT conditions give the diagonal-scaling form
with kernel `K = exp(-C/tau)`. Strict positivity and bounded costs avoid
degenerate zero entries, so unrolled-iteration arguments give
differentiability.

What this proves: the PT-ETB transport operator is a stable,
differentiable, label-independent neural layer for current-board
occupancy distributions.

What it does not prove: that puzzle-likeness is encoded in this transport
layer. That is the empirical hypothesis the benchmark is meant to test.

## Proposition 2: Central Falsification

Let `N(X)` collect material counts by side and type, side-to-move,
source-square marginal, target-square marginal, and capture-value
histograms derivable without move generation. Let `T(X)` be PT-ETB's true
transport summaries and let `T_rand(X)` be cost-shuffled summaries that
preserve each row's cost histogram but destroy target-square semantics.
If

```text
P(Y | N(X), T(X)) = P(Y | N(X), T_rand(X))
```

almost surely, then any classifier built on either feature set has the
same Bayes risk; the proposed transport geometry would carry no
label-relevant information beyond nuisance shortcuts. A measured gap
between PT-ETB and the cost-shuffled ablation is therefore evidence (not
proof) that the transport geometry carries useful signal.

## Counterexamples

- Quiet endgame studies driven by zugzwang or opposition rather than
  immediate piece-target geometry.
- Defensive or stalemate motifs not captured by transporting active
  pieces toward valuable targets.
- Splits where puzzle labels are dominated by material imbalance, source
  collection style, or non-geometric metadata absent from the tensor.
- Multi-ply tactical lines invisible from current occupancy alone.
- Ambiguous near-puzzles (fine label `1`) that intentionally sit between
  ordinary positions and true puzzles.

## Self-Critique

The strongest objection is that the model could learn a polished
material/proximity shortcut. The cost-shuffle ablation, material-only
nuisance ablation, matched-capacity CNN ablation, and type/geometry
removals exist to expose that failure. PT-ETB is still worth running
because it is small, label-safe, mathematically distinct from the
imported sheaf and move-delta families, and falsifiable through a single
randomized operator ablation.

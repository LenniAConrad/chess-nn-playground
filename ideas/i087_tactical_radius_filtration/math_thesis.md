# Math Thesis

Tactical Radius Filtration (`TRF`).

Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0857_tuesday_new_york_tactical_radius.md`.

Working thesis: a chess puzzle is multiscale in **tactical radius**, not in
image resolution. Scale `r` is defined as the minimum number of chess-rule
contact steps required to connect two squares, where contacts are typed
adjacencies derived from the current-board pseudo-legal attack/defense/ray
graph.

Let `S = {0, ..., 63}` and let `B` be the board decoded from the input
tensor. For each rule-relation type `t` in a finite set `T` (pawn attacks per
side, knight, bishop / rook / queen ray contacts, king contacts, friendly
defense, enemy attack, slider blocker, slider x-ray after one blocker, king
zone for each side, pawn front / promotion lane), define a deterministic
board-dependent adjacency `A_t(B) in {0,1}^{64 x 64}` whose nonzero entries
encode squares connected by relation `t` under current chess rules and
occupancy.

Form the untyped tactical contact graph

```
M(B) = I OR union_t A_t(B) OR A_t(B)^T
```

and define chess tactical distance

```
d_B(u, v) = min { r >= 0 : (M(B)^r)[u, v] = 1 }
```

using Boolean matrix multiplication. From this define closed tactical balls
and exact tactical shells

```
P_r(B) = 1[d_B(.,.) <= r]
Q_0(B) = I
Q_r(B) = P_r(B) AND NOT P_{r-1}(B), r >= 1.
```

Grouped shells `Q_{r,g}` arise from a coarsening map `pi_r : T -> G_r` that
unions relation types into typed bundles whose granularity decreases with
`r` (direct contacts at radius 1; attack/defense chains, ray continuations,
king-zone pressure at radius 2; king-pressure, material-tension, escape,
promotion, open-line complexes at radius 3). Each shell is row-normalized,
then used to mix per-square features `H_0` into shell-aggregated states
`H_r`. Each `H_r` is computed from `H_0` and `Q_r`, never from `H_{r-1}`, so
multiscale structure lives in the filtration `{Q_r}` rather than in network
depth.

The thesis is falsifiable: fine-2 puzzle positives should show systematically
different exact-shell signatures than fine-0/1 negatives, especially in the
radius-2 and radius-3 bundles around kings, high-value pieces, blockers, and
defended attackers. The packet's A1–A10 ablations (radius limit, exact vs
closed ball, chess vs Chebyshev graph, relation type shuffle, no x-ray, no
king zone, no shell counts, MLP/CNN baselines, forbidden-feature audit) are
the falsifiers; the architecture exposes the corresponding switches.

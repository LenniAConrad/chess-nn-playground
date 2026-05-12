# Math Thesis

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0812_tuesday_los_angeles_king_cage_dp.md`.

The thesis is that puzzle-like positions can contain an asymmetric king-cage signal: one king may be separated from the broader board by a current-position barrier made of attacks, blockers, edge geometry, and side-to-move pressure. The model exposes this signal through a differentiable soft shortest-path bottleneck rather than asking a CNN to infer escape topology implicitly.

## Formal Object

For the supported `simple_18` encoding, the adapter decodes current pieces:

```text
P(x) in {0,1}^{2 x 6 x 8 x 8}
```

and side-to-move. From the current board only, it computes pseudo-legal attack pressure `A_c(i)` for each color `c`. Sliding attacks stop at the first occupied blocker. No legal move generation, checkmate/stalemate oracle, engine evaluation, candidate move set, or future move-tree feature is used.

For defender color `c`, the barrier module produces:

```text
b_c(i) = softplus(base)
       + softplus(w_attack) * log1p(A_opponent(i))
       + softplus(w_own) * O_c(i)
       + softplus(w_opp) * O_opponent(i)
       + softplus(h_theta(local_features_i))
```

The monotone coefficients ensure that attack and occupancy evidence contributes nonnegatively to path cost.

## Soft Bellman-Ford Cage Energy

Let `G` be the fixed 8-neighbor king-step graph on the 64 board squares. For king square `k_c` and radius `r`, define the absorbing target shell:

```text
T_r(k_c) = { i : d_infty(i, k_c) >= r }
```

For temperature `tau > 0`:

```text
V_0(i) = 0 if i in T_r(k_c), else M
V_{t+1}(i) = 0 if i in T_r(k_c)
V_{t+1}(i) = -tau * logsumexp_{j in N_8(i)}(-(V_t(j) + b_c(j)) / tau) otherwise
```

The cage energy is:

```text
E_{c,r,tau}(x) = V_T(k_c)
```

The implementation computes this for multiple radii and temperatures, then forms side-to-move-relative features, opponent-minus-own cage gaps, extrema, and a temperature-spread proxy for path multiplicity.

## Proposition

For fixed barrier `b_c`, radius `r`, finite horizon `T`, and finite temperature `tau`, the recurrence computes a smooth soft minimum over bounded board paths from a square to the target shell. As `tau` approaches zero, the value at the king approaches the bounded-horizon minimum cumulative entered-square barrier among those paths.

The proof is the standard dynamic-programming induction: the base value represents an absorbing target shell, and each recurrence step expands one neighbor transition while log-sum-exp aggregates the exponential family of path extensions.

## Hypothesis Under Test

The classifier should benefit from the cage energy and distance fields when puzzle-likeness is tied to mating nets, trapped kings, or overloaded defenses near a king. The strongest falsifier is the degree-preserving random-grid DP: if replacing true board topology with a fixed random neighbor table gives the same result, the model is using attack/occupancy counts rather than corridor topology.

The repository task contract remains `puzzle_binary`: fine labels `0` and `1` are non-puzzle, and fine label `2` is puzzle. Fine labels and source metadata are diagnostics only and never neural inputs.

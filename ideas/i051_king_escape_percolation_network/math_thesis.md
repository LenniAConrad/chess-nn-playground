# Math Thesis

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0811_tuesday_pacific_king_escape_percolation.md`.

The core claim is that some puzzle-like positions contain a current-board tactical cage around one king. A local CNN can see attacked squares and blockers, but it is not forced to measure whether low-cost squares connect into an escape corridor. This idea exposes that structure directly through a bounded, differentiable soft shortest-path operator over the king-adjacency grid.

## Formal Object

For `simple_18`, the adapter decodes current-board piece occupancy:

```text
P(x) in {0,1}^{2 x 6 x 8 x 8}
```

with colors `{white, black}` and piece types `{pawn, knight, bishop, rook, queen, king}`. For each side `s`, pseudo-legal attack counts `A_s(x, v)` are computed from frozen-board attack geometry only. Sliding attacks stop at the first occupied blocker; no legal move generation, checkmate oracle, or engine signal is used.

For defender side `s`, the model learns a nonnegative cell cost:

```text
c_s(x, v) = softplus(g_theta(phi_s(x, v))) + base_cost + lambda_occ * occ_without_king_s(x, v)
```

where `phi_s` contains defender/attacker piece planes, attacker attack type counts, defender defense count, king distance, edge distance, and side-to-move role bits.

## Soft Escape Recurrence

Let `G_K` be the 8-neighbor king-adjacency graph with self-loops on the `8x8` board. For temperature `tau` and horizon `T`, the implemented recurrence is:

```text
D_0(v) = 0 if v = K_s(x), else large_value
D_{t+1}(v) = c_s(x, v) - tau * logsumexp_{u in N_K(v)}(-D_t(u) / tau)
```

The edge escape free energy is a soft minimum over edge squares:

```text
F_edge = -tau * logsumexp_{v on board edge}(-D_T(v) / tau)
```

The model also exports outer-ring free energies and reachable masses:

```text
M_alpha = mean_v sigmoid((alpha - D_T(v)) / rho)
```

These form a bottleneck vector, while saved reachability maps are fused with a small board stem.

## Proposition

For fixed cost field `c_s`, temperature `tau > 0`, and horizon `T`, the recurrence computes the path free energy:

```text
D_T(v) = -tau * log sum_{p in P_T(K_s, v)} exp(-sum_i c_s(p_i) / tau)
```

where `P_T(K_s, v)` is the set of length-`T` frozen-board king paths from the defender king to `v`. As `tau -> 0`, `D_T(v)` approaches the ordinary bounded-horizon minimum path cost for squares with at least one length-`T` path.

## Hypothesis Under Test

The classifier should improve when it receives direct escape free energies and reachable masses, especially in positions where the side to move has constrained the opponent king's escape corridors. The main falsifier is to preserve ring, occupancy, and coarse hazard-bin marginals while disrupting connected corridors before the DP. If that shuffled-cost control matches the full model, then this operator is likely exploiting static density shortcuts rather than percolation structure.

The implementation keeps the repo's `puzzle_binary` target contract: fine labels `0` and `1` are non-puzzle, and fine label `2` is puzzle. Fine labels and source metadata are never model inputs.

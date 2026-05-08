# Math Thesis

Hall-Defect Zeta Operator (HDZ).

Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0802_tuesday_new_york_hall_defect_zeta.md`.

## Working thesis

A true tactical puzzle is more likely than a near-puzzle to contain a small local set of tactical obligations whose effective defender neighborhood has positive Hall defect after pins and king-exposure constraints are applied. Equivalently, for some side `c`, anchor square `t`, and obligation subset `U ⊆ Ω_{c,t}`,

    |U| > |N(U)|,

where `N(U) = ⋃_{o ∈ U} N(o)` is the union of effective defenders covering at least one obligation in `U`. By Hall's theorem, a positive deficiency `δ(U) = |U| − |N(U)|` certifies that no distinct-responder assignment can cover the obligations in `U`. Near-puzzles often repair the inequality by adding exactly one defender, unpinning a defender, changing a ray blocker, or providing a king escape resource; HDZ is designed to expose those repairs as algebraic differences.

## Finite-algebraic objects

For each side `c` and anchor square `t`:

- `Ω_{c,t}` is a deterministically ordered set of at most twelve obligation atoms (anchor, king ring within Chebyshev 2, high-value pieces within distance 2, attacked pieces within distance 3, line-interposition squares on opponent slider lines through at most one blocker, then remaining Chebyshev rings).
- `P_c` is the set of pieces of color `c`.
- `R_{c,t}(o, p) = 1` exactly when `p` legally and effectively answers obligation `o` after a pin and king-exposure filter is applied to the raw contact relation. Pinned pieces are restricted to their pin line, and the king cannot defend squares occupied by friends or controlled by the opponent.

The aggregation step is the Boolean-lattice zeta union `D(U) = ⋃_{o ∈ U} D_o` over atom neighborhoods, restricted to subsets of order `r ∈ {1,2,3,4}`. The HDZ tensor reports per-order `max`, `mean`, `mindefenders`, and `pinshare` summaries together with four anchor-local scalar channels.

## Outputs

The model returns one logit `z = f_θ(X, HDZ(X))`, trained with weighted BCE on `y = 1[fine_label = 2]`. HDZ is deterministic and not differentiated through. The packet's controls — AtomScramble-HDZ, NeuralSynth-40, the no-pin-filter ablation, subset-order trims, and obligation-universe trims — are exposed as configuration switches on the same trainable network.

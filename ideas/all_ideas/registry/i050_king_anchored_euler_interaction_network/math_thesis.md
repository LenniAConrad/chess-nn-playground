# Math Thesis

King-Anchored Euler Interaction Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0809_tuesday_local_euler_interaction.md`.

## Setup

For a current-board encoding `x ∈ ℝ^{18 × 8 × 8}` (`simple_18`), define
side-relative role bitboards `roles[r] ∈ {0, 1}^{8 × 8}` for
`r ∈ {own_pawn, own_minor, own_heavy, own_king, opp_pawn, opp_minor, opp_heavy,
opp_king}` using only piece occupancy and side-to-move. Let `Q` be the cubical
complex whose 2-cells are the 64 unit board squares and whose 1-/0-cells are
the grid edges and vertices.

Anchors: `A(x) = {opp_king(x), own_king(x), centre = (3.5, 3.5)}`. Directions:
the eight king directions `U = {(±1,0), (0,±1), (±1,±1)}`. Thresholds `τ` are
sampled on `[-7, 7]` (default 15 evenly-spaced values).

For role `r`, anchor `a`, direction `u`, and threshold `τ`, define the swept
role subcomplex

```
K_{r,a,u,τ}(x) = cubical_closure({c ∈ Q_2 : roles[r](c)=1 and ⟨u, coord(c) - a⟩ ≤ τ}).
```

The role Euler curve and its first difference are
`E_{r,a,u}(τ) = χ(K_{r,a,u,τ})` and `ΔE_{r,a,u}(τ) = E(τ_{i+1}) - E(τ_i)`.

For each role pair `(r, s)` in the default 8-pair list (heavy/minor/pawn ×
opposing king, plus heavy×heavy and minor×heavy contacts), the Euler
interaction curve is

```
J_{r,s,a,u}(τ) = χ(K_r ∪ K_s) - χ(K_r) - χ(K_s).
```

## Proposition

By finite additivity of Euler characteristic on finite cell complexes,
`χ(A ∪ B) = χ(A) + χ(B) - χ(A ∩ B)`, hence
`J_{r,s,a,u}(τ) = -χ(K_r ∩ K_s)`. Because distinct role bitboards cannot share
a 2-cell, nonzero `J` corresponds to topological contact along shared edges,
shared vertices, or merging/enclosure events induced by the cubical closure.

The bespoke implementation computes `χ` directly as `V - E + F` of the closure
and so realises the proposition exactly on hard binary masks.

## Hypothesis

Puzzle-like positions exhibit sharply organised swept contact, enclosure, and
separation events between role-defined regions around the kings; therefore the
king-anchored interaction curves carry signal beyond material and naive
proximity histograms. The optimisation objective is balanced cross-entropy on
the puzzle_binary target; trainable parameters live only in the MLP head, so
the model is constrained to use the topological observable rather than learned
local filters.

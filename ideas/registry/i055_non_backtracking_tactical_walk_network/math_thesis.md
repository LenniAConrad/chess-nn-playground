# Math Thesis

Non-Backtracking Tactical Walk Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0922_tuesday_local_nonbacktracking_walk.md`.

## Working thesis

Puzzle-like positions are disproportionately marked by short, directed chains of
current-board tactical dependency in which attack/protection pressure propagates
without immediately undoing itself. A Hashimoto-style non-backtracking edge-walk
operator should expose that signal while suppressing trivial reciprocal-attack and
degree/material shortcuts.

## Formal object

For board `x`, let `G_x = (V_x, E_x, tau_x)` be the directed typed edge graph where
each occupied source piece projects edges to:

- enemy occupied targets it attacks (`enemy_piece_attack`),
- own occupied targets it protects (`friendly_piece_protect`),
- virtual enemy king-zone squares it attacks (`enemy_king_zone_attack`),
- virtual own king-zone squares it protects (`own_king_zone_protect`).

The unweighted Hashimoto non-backtracking operator on directed edges is

```text
H_x[e, f] = 1{ terminal(e) = origin(f) } * 1{ origin(e) != terminal(f) }
```

restricted to edges whose terminal is an occupied node (virtual king-zone targets
have no outgoing transitions). The typed weighted operator used by the model is

```text
~H[e, f] = H_x[e, f] * a_theta(tau(e), tau(f))
```

with `a_theta` realised as `W_shared + sum_j alpha[type_pair, j] W_j` plus a
per-target-relation bias.

## Proposition (recapped)

`(H_x^k)[e, f]` equals the number of length-`k+1` directed edge walks from `e` to
`f` that satisfy the non-backtracking condition at every step. Reciprocal
oscillations `u → v → u` contribute zero. The neural recurrence implemented in
`NonBacktrackingEdgeBlock` is a learned, typed, nonlinear relaxation of these
walk counts; it does not prove label relevance, only what walks the operator
counts.

## Hypothesis

`I(Y; Psi_K(H_X, E_X) | N(X)) > 0` for small `K`, where `N(X)` collects nuisance
statistics (material, side-to-move, edge counts, degree histograms,
relation-pair counts) and `Psi_K` is the pooled non-backtracking edge-walk
representation. The central falsification ablation is the
`randomized_transitions` mode, which permutes destination edges inside each
`(rel(e), rel(f))` bucket; if it matches the main model, the non-backtracking
continuation relation is not the source of signal.

## Implementation

This thesis is implemented as a bespoke PyTorch model. See
`ideas/registry/i055_non_backtracking_tactical_walk_network/architecture.md` for the
implementation binding (registered model name, source file, idea wrapper) and
the forward contract.

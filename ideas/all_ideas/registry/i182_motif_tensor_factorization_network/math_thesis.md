# Math Thesis

Motif Tensor Factorization Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `6`.

## Working Thesis

Puzzle signal is often a *multiplicative* relation among typed roles:

```
attacker type x target type x defender state x line relation x tempo
```

A plain CNN learns these implicitly; we instead represent each typed
role candidate explicitly and score the conjunction through a low-rank
CP factorization on a 4-way motif tensor. The top motifs, their
entropy, and a near-disproof leg-strength score give the puzzle head
features that *cannot* be made strong unless every conjunction
component is also strong.

## Motif Tensor

For attacker, target, defender candidate sets indexed by `i, j, k`
respectively, and a line-relation `R_ij` between attacker `i` and
target `j`, the motif tensor `M ∈ R^{P × P × P}` is

```
M[i, j, k] = sum_{r=1..R} A_r(i) * T_r(j) * D_r(k) * rel_r(i, j)
```

where `A_r, T_r, D_r ∈ R^{P × R}` are role-specific CP factors and
`rel_r ∈ R^{P × P × R}` is a learned line-relation factor. CP
factorization keeps the parameter count linear in `R` rather than
quartic.

## Pooling

The classifier reads four motif statistics:

- `top_motif_scores`: top-`top_motifs` values of the flattened motif
  tensor.
- `motif_entropy`: `-sum p log p` over `softmax(M)`, low when one
  conjunction dominates.
- `motif_contrast`: difference of mean top-motif scores between the
  actual position and a side-to-move-flipped intervention, capturing
  the "tempo" leg of the conjunction.
- `near_disproof_score`: smallest per-leg magnitude of the best
  motif's CP factors — the multiplicative motif's weakest leg.

## Ablations

- `additive_motif_score`: replace `A * T * D * rel` with
  `A + T + D + rel`. The additive form cannot collapse on a missing
  leg, so it should perform measurably worse if the multiplicative
  conjunction is doing the work the thesis predicts.
- `no_relation_embedding`: replace `rel` with ones. Tests whether
  chess geometry between attacker and target is contributing.
- `rank_8_24_64`: sweep CP rank `R` to measure capacity.

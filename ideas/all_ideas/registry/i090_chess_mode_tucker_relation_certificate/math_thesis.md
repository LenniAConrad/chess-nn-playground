# Math Thesis

Chess-Mode Tucker Relation Certificate (`CMTRC`).

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0900_tuesday_new_york_relation_tucker.md`.

The binary puzzle decision is modeled as a Tucker contraction over a fixed
chess-relation moment tensor:

\[
\Pr(y=1 \mid x) = \sigma\left( g\left(
  \left\{\langle T(x) \times_K U_K^\top \times_R U_R^\top
                  \times_D U_D^\top \times_G U_G^\top,\, \Omega_h \rangle
  \right\}_{h=1}^{H}
\right) \right).
\]

The relation tensor is built from fixed legal-chess relation masks and
deterministic board-region masks, not learned attention. Concretely:

- `E_{b,k,s}` is the latent board embedding produced by a `1x1` channel
  lift, a GroupNorm, and a SiLU.
- `M_{\rho,\delta,s,t}` is the fixed sparse relation mask covering 8 sliding
  rays, signed knight jumps, king-adjacent moves, and the white / black
  pawn-attack geometries, with a depth axis of size 8 indexing 1..7 ray
  distance and signed jump variants.
- `A_{\gamma,s}` is a deterministic, normalised board-region mask covering
  the full board, light/dark squares, the four-square center, the extended
  16-square center, the corners, the edges, the back ranks, the side files,
  and the promotion bands.

The relation scan and moment tensor are

\[
N_{b,k,\rho,\delta,s}
= \sum_t M_{\rho,\delta,s,t}\, E_{b,k,t},
\quad
T_{b,k,\rho,\delta,\gamma}
= \sum_s A_{\gamma,s}\, E_{b,k,s}\,
  \tanh\!\left(\frac{N_{b,k,\rho,\delta,s}}{\sqrt{\deg(\rho,\delta,s)+\epsilon}}\right).
\]

The Tucker mode projection
`S_b = T_b \times_K U_K^\top \times_R U_R^\top \times_D U_D^\top \times_G U_G^\top`
produces a small `(rK, rR, rD, rG) = (8, 6, 4, 5)` core. Contracting against
the learnable Omega tensor yields the `H = 24` hidden vector that feeds a
two-layer head. The thesis is that the binary puzzle target is well predicted
by low-multilinear-rank interactions among the **piece-state channel**,
**lawful chess relation**, **distance / jump**, and **board-region** modes,
without learned square-pair attention or generic CNN/Transformer mixing.

The thesis is falsified by the same-parameter non-tensor control
**FlatProjectedMLP**, which uses the same stem and the same fixed relation
tensor but flattens `T` through a deterministic signed CountSketch into 112
features and feeds them to a parameter-matched MLP head. If the control
matches CMTRC across seeds, the chess-mode Tucker structure is not earning
its complexity. The model also reports a per-example multilinear rank
certificate so collapses (`eff_rank ≈ 1`) and saturations (`eff_rank ≈ max`)
are visible in the metric stream.

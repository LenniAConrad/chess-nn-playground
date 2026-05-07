# Math Thesis

Attack-Defense Sheaf Energy Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0255_tuesday_local_attack_defense_sheaf.md`.

Working thesis: Puzzle-likeness in a chess position is correlated with high,
structured sheaf residual energy on a dynamic typed chess-incidence complex.
A puzzle-like position often contains local attack-defense constraints that
cannot be made mutually consistent: pinned defenders, target-square
convergence of multiple attackers, ray attacks through blockers, or
overloaded shields. A typed sheaf coboundary

```text
c_e = sqrt(gamma_e) * (R_src^tau h_u - R_dst^tau h_v)
```

over rays, knight jumps, king-neighborhood edges, and oriented pawn-attack
diagonals directly parameterizes that local inconsistency. Ray edges are
gated by a learned occupancy-proxy visibility product
`q_e = prod_{m in M_e} (1 - o_m + eps)` so the model learns pseudo
line-of-sight without consulting an engine or legality oracle. Sheaf energy
is summed at the edge level and pooled at the destination square (target
convergence) to drive the binary puzzle classifier.

The model is trained as one BCE-with-logits logit head per square-tensor input
`(B, C, 8, 8) -> (B,)` with the puzzle_binary mapping (fine label `0` ->
non-puzzle, fine labels `1` and `2` -> puzzle). CRTK/verification/source
metadata is never consumed as model input.

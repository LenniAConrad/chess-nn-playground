# Math Thesis

Geometry-Conditioned Board Pseudo-Likelihood Ratio Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0713_tuesday_local_geom_plr.md`.

The thesis is that puzzle-like positions should have a different static board-token dependency structure from non-puzzles. Instead of pooling CNN features directly into a discriminative head, GeomPLR learns two class-conditioned pseudo-likelihood models over the current board:

```text
q_c(t_i | t_N(i), m, i)
```

where `t_i` is the 13-way token on square `i`, `t_N(i)` are leave-self-out neighbors selected only by static chess geometry, and `m` is metadata from the board tensor: side-to-move, castling rights, and en-passant summary.

For each class `c`, the model computes a weighted pseudo-description length:

```text
S_c(t, m) = sum_i w(t_i) * CE(q_c(t_i | t_N(i), m, i), t_i)
```

Empty squares use a smaller weight so the score is not dominated by empty-board reconstruction. Scores are normalized by total board weight. The packet classifier is the pseudo-log-likelihood ratio:

```text
z_c = -S_c / softplus(temperature) + b_c
```

The implemented repo head preserves that mechanism while adapting it to the configured puzzle-binary trainer. The trainer expects one BCE logit where fine labels `0` and `1` are non-puzzle and fine label `2` is puzzle, so the model returns:

```text
logit = z_1 - z_0
```

The class-1 decoder is therefore trained as the puzzle-side pseudo-likelihood model under the repository label mapping. No engine output, legal move enumeration, verification status, CRTK tag, source field, or future-line information is used as input.

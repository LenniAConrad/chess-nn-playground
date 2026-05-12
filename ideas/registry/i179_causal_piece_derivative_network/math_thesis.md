# Math Thesis

Causal Piece-Derivative Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `3`.

Working thesis: In true puzzles, the puzzle signal often depends
sharply on a few critical pieces or squares. In near-puzzles, the
score may come from broad tactical texture without a decisive
dependency.

The implementation realises this as a *causal piece-derivative*
readout. A trunk produces a base logit and per-square features; a
gating head selects the top-`K` candidate squares; for each candidate
the model applies deterministic interventions (`remove_piece`,
`hide_square`, `neutralize_side`) through a lightweight shared delta
encoder, and the puzzle logit is

```
sensitivity_{i, t} = base_logit - delta_logit_{i, t}
puzzle_logit       = base_logit
                   + criticality_mlp(
                         max, top2_gap, entropy, signed_sum,
                         own_vs_enemy_split)
```

over the candidate sensitivities. Real puzzles should have a
peaked criticality profile — a few pieces dominate the sensitivity
field — while near-puzzles should have flat sensitivities even when
the trunk is confident.

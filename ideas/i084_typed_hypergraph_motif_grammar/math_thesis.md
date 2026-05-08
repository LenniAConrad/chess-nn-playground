# Math Thesis

Typed Hypergraph Motif Grammar

Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0757_tuesday_new_york_motif_grammar.md`.

Working thesis: Select **Typed Hypergraph Motif Grammar** as the
research direction. A board state is parsed into a typed hypergraph
whose nodes are pieces and squares and whose typed hyperedges encode the
deterministic relations a chess solver must reason over: `attacks_piece`,
`defends_piece`, `attacks_square`, `same_color`, `opp_color`,
`king_zone`, `near_king_piece`, `slider_aligned`, `pinned_to_king`,
`only_blocker_between`, `loose_piece`, `underdefended_piece`,
`high_value_target`, and `king_piece`.

Over this typed hypergraph we define a small motif grammar with
productions:

```
pressure(a, t)              :- attacks_piece(a, t), opp_color(a, t).
loose_target(a, t)          :- pressure(a, t), loose_piece(t).
loose_target(a, t)          :- pressure(a, t), underdefended_piece(t).
king_zone_pressure(a, k, s) :- attacks_square(a, s), king_zone(k, s),
                              king_piece(k), opp_color(a, k),
                              near_king_piece(a, k).
pin_shape(p, b, k)          :- pinned_to_king(p, b, k).
line_pressure(s, b, t)      :- only_blocker_between(b, s, t),
                              slider_aligned(s, t).
fork_shape(a, t1, t2)       :- pressure(a, t1), pressure(a, t2),
                              t1 != t2, high_value_target(t2).
battery_shape(s1, s2, t)    :- same_color(s1, s2),
                              only_blocker_between(s2, s1, t),
                              line_pressure(s1, s2, t).
compromised_defender(d, t, k) :- defends_piece(d, t),
                                 exists p. pin_shape(p, d, k).
overload_shape(d1, d2, t1, t2) :- defends_piece(d1, t1),
                                  defends_piece(d2, t2),
                                  loose_target(_, t1),
                                  loose_target(_, t2),
                                  d1 != d2.
tactical_convergence(*)     :- loose_target * compromised_defender_by_king
                                | pressure * overload_by_target * king_zone_by_king.
puzzle_like_motif(*)        :- tactical_convergence * king_zone_by_king
                                | fork_shape with high-value king target.
```

Productions are evaluated as masked `logsumexp` / `logaddexp` over the
typed hypergraph, with a learned `production_bias` per rule. Each pair
of pieces carries a learned pair score that combines piece-attribute
embeddings and the typed relation features. Per-production chart
statistics (max, logsumexp, mean, log-count, soft density) feed both a
fused readout over the pooled board features and a grammar-only
ablation head.

The puzzle logit is read out from the `logsumexp` of the highest-level
productions composed with the lower-level evidence, so the grammar's
hierarchical structure --- not a flat per-square CNN response --- is the
decisive non-linearity used to rank puzzle-likeness. A
`grammar_depth` switch caps how many composition layers contribute,
giving us a clean depth ablation that isolates the role of the
chart-parsed grammar above the convolutional trunk.

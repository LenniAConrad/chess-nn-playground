# Math Thesis

Forcing-Certificate Transformer

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `1`.

Working thesis: A real puzzle should admit a compact tactical
certificate

```
attacker / forcing piece
target
defender / escape resource
blocker / pin / overload relation
tempo side
```

so a position should be classified through a small set of structured
certificate slots, not only a single global board embedding.
Near-puzzles often look tactical but fail because the certificate has
a hole, so slot competition (with chess relation priors) gives the
classifier a way to refuse positions whose certificate does not
close.

The implementation realises this by introducing `K` learnable slot
queries that cross-attend to 64 square tokens with chess relation
biases (same line, knight reach, king-zone adjacency, pawn attacks),
emitting per-slot scores that are aggregated as

```
puzzle_logit = logsumexp(slot_score_k) + global_residual_logit
```

`logsumexp` lets a few highly-confident slots drive the puzzle logit,
and the residual head keeps the model competitive on positions whose
certificate is distributed.

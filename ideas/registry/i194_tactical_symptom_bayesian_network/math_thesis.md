# Math Thesis

Tactical Symptom Bayesian Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `9`.

Working thesis: Many tactical concepts behave like noisy logical
symptoms — *king exposed*, *defender overloaded*, *line opened*,
*piece pinned*, *queen aligned*, *escape squares reduced*, *target
under-defended*. A real puzzle is the *conjunction* of compatible
symptoms. A near-puzzle activates some of them but misses the
required cluster.

So we make the symptoms explicit and combine them through a
differentiable Bayesian-style noisy-AND/noisy-OR network instead of
an uninterpretable dense pool. With

- `K` learned per-square sigmoid symptom heads, lifted to image-level
  probabilities by a noisy-OR over squares
  `s_k = 1 - prod_sq (1 - s_k_sq)`,
- `J` latent causes obtained by a noisy-OR over symptoms with
  non-negative weights and a per-cause leak
  `cause_j = 1 - (1 - leak_j) * prod_k (1 - w_jk * s_k)`,
- a learned mixture of noisy-OR and noisy-AND aggregations of the
  causes, `puzzle_prob = alpha * prob_or + (1 - alpha) * prob_and`,

the output is the source-packet rule

```text
puzzle_logit = logit(clamp(puzzle_prob)) + residual_weight * residual_logit
```

where `residual_logit` is a small MLP read from pooled features +
symptoms + causes. The expected gain over generic CNN pooling is
that near-puzzles are forced to activate a *consistent* symptom
cluster before producing a strongly positive logit.

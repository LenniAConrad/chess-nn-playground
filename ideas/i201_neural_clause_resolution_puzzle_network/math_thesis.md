# Math Thesis

Neural Clause-Resolution Puzzle Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`.

Batch candidate rank: `4`.

## Working Thesis

A puzzle often follows from a small proof made of typed facts:

```text
Attack(piece, target)
Defends(piece, target)
Pinned(piece, king)
LineOpen(piece, target)
EscapeSquare(square)
Tempo(side)
```

Soft Horn clauses such as

```text
PuzzleWitness(X, Y, Z) :- Attack(X, Y), Pinned(Z), Defends(Z, Y), Tempo(us)
```

derive `PuzzleWitness` from a small conjunction of typed facts with
shared variables. Real puzzles tend to admit such a derivation; near
puzzles tend to fail at the unification or conjunction step. A
differentiable clause-resolution layer is the natural object to test
this directly.

## Formal Object

For each board `x` we build:

- A learnable predicate embedding table
  `predicate_embeddings ∈ R^{P × d}`. The first `P_u` rows are unary
  square-valued predicates (`attack`, `defends`, `pinned`, `line_open`,
  `escape_square`); the last `P_g` rows are global predicates
  (`tempo`).
- An initial unary fact base `F_0(x) ∈ [0, 1]^{P_u × S}` produced by a
  per-predicate `1×1` convolution over the trunk feature map.
- An initial global fact bank `G_0(x) ∈ [0, 1]^{P_g}` produced by a
  linear readout of the pooled trunk summary.
- Per-clause head and body predicate queries
  `head_query[c] ∈ R^d`, `body_query[c, k] ∈ R^d`. Soft predicate
  selectors are
  `head_sel[c, p] = softmax_p(head_query[c] · predicate_embeddings[p])`
  and similarly for the body slots, implementing differentiable
  predicate choice.
- A bank of `R` row-stochastic spatial relation kernels
  `relations[r, s, s'] = softmax_{s'} relation_logits[r, s, s']`,
  selected per body slot by a soft mixture
  `body_rel[c, k, r]`. Composing `body_rel` and `relations` yields a
  differentiable variable-binding operator: a body predicate evaluated
  at the head's square `s` reads truth values at related squares `s'`.

## Iterated Resolution

A single resolution step computes, for each clause `c`, the soft
conjunction of its body slots and projects the result onto the head
predicates:

```text
F_rel[b, r, p, s]    = sum_{s'} relations[r, s, s'] * F_unary[b, p, s']
rel_mixed[b, c, k, p, s] = sum_r body_rel[c, k, r] * F_rel[b, r, p, s]
body_score[b, c, k, s]   = sum_{p < P_u} body_sel[c, k, p] * rel_mixed[b, c, k, p, s]
                         + sum_g       body_sel[c, k, P_u + g] * G[b, g]
clause_activation[b, c, s] = sum_k log(body_score[b, c, k, s] + eps) + clause_bias[c]
clause_truth[b, c, s]      = sigmoid(clause_activation[b, c, s])
```

The fact base is updated by a residual probabilistic-OR with a
per-predicate gate `gate ∈ [0, 1]`:

```text
delta_unary[b, p, s] = sum_c head_sel[c, p < P_u] * clause_truth[b, c, s]
F_{t+1}[b, p, s]     = F_t[b, p, s] + (1 - F_t[b, p, s]) * gate[p] * delta_unary[b, p, s]
```

with the analogous update for `G`. This iteration is applied
`K = resolution_rounds` times, producing the trajectories
`(F_0, F_1, …, F_K)` and `(G_0, G_1, …, G_K)`.

## Decision Rule

A small head MLP maps the readout

```text
[ pool_mean(F_K), pool_max(F_K), G_K, board_pool, clause_summary ]
```

to one puzzle logit. Real puzzles concentrate clause activation into a
coherent set of head predicates (the soft `PuzzleWitness`); near
puzzles either fail to compose body predicates through the relation
kernels or fail to commit to a head, and that imbalance feeds the
classifier.

This is a bespoke implementation of the markdown thesis; it is no
longer a `ResearchPacketProbe` scaffold.

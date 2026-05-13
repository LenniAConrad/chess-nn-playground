# Math Thesis

Source: `ideas/research/primitives/codex_03_reply_channel_capacity.md`
(Reply Channel Capacity Primitive, RCC).

## Working thesis

For a position `x`, let `L(x) in R^{K x R}` be a learned candidate /
reply logit table. Each row is a soft conditional distribution over
replies given a candidate:

```
P_{ij} = P(reply = j | candidate = i) = softmax_j(L_{ij} / tau)
```

Channel capacity asks for the maximum mutual information between the
candidate choice and the reply distribution:

```
q*       = argmax_q I_q(I; J)
r_j      = sum_i q_i P_{ij}
I_q(I; J) = sum_i q_i sum_j P_{ij} log (P_{ij} / r_j)
```

solved by damped Blahut-Arimoto iterations. The primitive returns
capacity (nats and bits), the capacity-achieving candidate prior
`q*`, the reply marginal `r`, the conditional entropy
`H(reply | candidate)` weighted by `q*`, the output entropy `H(r)`,
the capacity gap `H(r) - H(reply | candidate)`, and per-row reply
entropies.

The architecture-level claim is additive:

```
final_logit(x) = i193_trunk(x) + gate(x) * delta(x)
```

with `delta(x), gate(x)` small MLPs over `[q*-pooled cand,
r-pooled reply, trunk_pool, capacity, capacity_bits,
conditional_entropy, output_entropy, capacity_gap]` and
`[trunk_pool, capacity, capacity_gap, conditional_entropy]`
respectively.

## Why this matters

Entropy alone can be fooled by one sharp decoy candidate or by a
broad reply distribution that is still highly candidate-dependent.
RCC asks the sharper question: how much can the candidate choice
*control* the reply distribution? For real forcing tactics, strong
candidates induce distinct reply distributions, capacity is high. For
near-puzzles, the candidate choice barely changes the reply
landscape, capacity collapses. The capacity gap and conditional
entropy together form a richer signal than `min_i H(reply | i)` or
`mean_i H(reply | i)`.

## Falsifier

- Primitive-level: shuffle each row's reply distribution (ablation
  `row_shuffle_channel`) or force all rows to the first row
  (`duplicate_rows`); capacity collapses to near zero. The
  `entropy_only` ablation feeds only `conditional_entropy` to the
  fusion head and zeros out everything else — tests whether the full
  capacity solution beats `i192`-style entropy diagnostics.
- Architecture-level: p003 must improve matched-recall near-puzzle FP
  at recall 0.80 by at least 2% over an entropy-only baseline on the
  same parent, without regressing aggregate PR AUC by more than 0.005.

## Composition with other Codex reply primitives

RCC is one of five candidate/reply primitives compiled from the
2026-05-12 Codex batch:

- p001 (PAFR): partial-order frontier over candidate utility table;
  ignores reply distributions.
- p002 (RSP): payoff-game saddle reducer; reads payoffs, not
  distributions.
- p005 (WCQ): nested adversarial quantifier; doesn't model the
  conditional reply distribution.

Together, p002, p003, and p005 are intentionally orthogonal views of
the candidate/reply table: RSP solves the game, RCC measures
information capacity, WCQ asks whether one witness survives.

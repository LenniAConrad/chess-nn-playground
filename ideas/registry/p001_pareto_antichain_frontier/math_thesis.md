# Math Thesis

Source: `ideas/research/primitives/codex_01_pareto_antichain_frontier.md`
(Pareto Antichain Frontier Primitive, PAFR).

## Working thesis

For a position `x`, let `U(x) in R^{K x C}` be a learned candidate
utility table with `K` candidate tokens and `C` utility channels
(e.g. forcing claim, king exposure, exchange soundness, reply safety,
promotion lane, defender overload). Existing aggregation operators
(scalar max, attention with scalar score, sparsemax, top-k) all
project `U` to a single ranking, losing the partial order. PAFR keeps
the partial order intact:

```
s_{ij}^c = sigmoid((U_{ic} - U_{jc} - eps_c) / tau_dim)
p_{ij}   = prod_c s_{ij}^c
log pi_j = sum_{i != j} log(1 - p_{ij})
pi_j     = exp(log pi_j)
alpha_j  = softmax_j((log pi_j + beta * q_j) / tau_set)
summary  = sum_j alpha_j V_j
width    = sum_j pi_j
entropy  = -sum_j alpha_j log alpha_j
```

with `q_j = mean_c U_{jc}`. As `tau_dim -> 0`, `pi_j` converges to
the exact non-dominated indicator under the product partial order.

The architecture-level claim is additive:

```
final_logit(x) = i193_trunk(x) + gate(x) * delta(x)
```

with `delta(x), gate(x)` small MLPs over `[summary, width, entropy,
trunk_pool]` and `[trunk_pool, width, entropy]` respectively. The
trunk runs once and supplies both the base logit and the spatial
features used to compile candidate tokens.

## Why this matters

The per-class puzzle_binary benchmark identifies near-puzzle false
positives at matched recall as the central pressure point. PAFR is
built for the case where no single scalar tactical score is
trustworthy: real puzzles should often have a narrow nondominated
frontier (one candidate beats the rest on every axis), while
near-puzzles often have a wide frontier (one candidate is good on
claim, another on safety, no candidate cleanly dominates). The
operator surfaces frontier width and entropy as first-class signals
that the trunk does not currently expose.

## Falsifier

- Primitive-level: shuffle the utility channels per candidate
  (ablation `shuffle_channels`) or collapse to a scalar max
  (ablation `scalar_max`) and verify both lose most of the lift on
  the target slices. The single-channel control (`single_channel`)
  must also be beaten — if it matches, the partial-order structure
  is not load-bearing and the primitive is dropped.
- Architecture-level: p001 must improve near-puzzle false positives
  at recall 0.80 by at least 3% over i193, without regressing
  aggregate PR AUC by more than 0.005.

## Composition with other Codex reply primitives

PAFR is one of five candidate/reply primitives compiled from the
2026-05-12 Codex batch. The four others are intentionally orthogonal:

- p002 (RSP): solves an entropy-regularized saddle game over a
  payoff table; PAFR ignores payoffs and operates on the partial
  order over utility channels.
- p003 (RCC): measures channel capacity of the reply distribution;
  ignores partial-order structure.
- p004 (TCC): measures tail-copula concordance across per-square
  evidence channels; PAFR is over candidates, TCC is over squares.
- p005 (WCQ): nested adversarial quantifier (exists candidate /
  forall counterwitness); not a partial-order reducer.

All five primitives share the BoardTokenAttention compiler and the
i193 trunk-feature helper to keep the comparison clean across heads.

# Math Thesis

Ray-Language Automaton Network (`RLAN`).

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0719_tuesday_local_ray_language_automaton.md`.

Thesis: chess puzzle-likeness is enriched for short, ordered, gapped
piece-token strings on the 92 oriented rays of the 8x8 board (ranks,
files, diagonals, anti-diagonals taken in both directions). Such
motifs (pins, skewers, batteries, back-rank alignments, etc.) are
regular languages over a 14-symbol side-relative alphabet, so a family
of differentiable weighted finite automata over the log semiring
should learn them more sample-efficiently than an unconstrained 2D
CNN.

## Operator

For each automaton `r = 1..R`, define the log-semiring WFA
`A_r = (alpha_r, {T_{r,a}}_a, omega_r)` with start weights
`alpha_r in R^Q`, symbol-conditioned transitions `T_{r,a} in R^{QxQ}`,
and final weights `omega_r in R^Q`. For a ray string
`s = (a_1,..,a_T)` the recurrence is

```
h_0(j) = alpha_r(j)
h_t(j) = logsumexp_i [ h_{t-1}(i) + T_{r, a_t}(i, j) ]
score_r(s) = logsumexp_j [ h_T(j) + omega_r(j) ]
```

A context-conditioned affine bias is added to each accept score:
`tilde_score_r(s_l, c_l) = score_r(s_l) + b_r(c_l)`. Board features
are pooled by global max/log-sum-exp and per-axis max/log-sum-exp;
those summaries plus safe board metadata feed a small MLP that returns
one puzzle logit for the BCE-with-logits puzzle_binary trainer.

## Falsification

The packet's central falsifier is a token-permutation ablation that
preserves material and metadata while destroying ray order. The
bespoke implementation is hyperparameter-clean: switching to permuted
tokens is a one-line change in the parser. If permuted-token training
matches the main run, ordered ray language is not driving the signal.

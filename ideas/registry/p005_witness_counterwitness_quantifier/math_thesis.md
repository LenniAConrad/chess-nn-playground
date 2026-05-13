# Math Thesis

Source: `ideas/research/primitives/codex_05_witness_counterwitness_quantifier.md`
(Witness-Counterwitness Quantifier Primitive, WCQ).

## Working thesis

For a position `x`, let `claim(x) in R^K` be per-candidate forcing
claim scores and `counter(x) in R^{K x R}` be per-(candidate, reply)
counterwitness scores. WCQ computes:

```
counter_envelope_i = tau_forall * logsumexp_j(counter_{ij} / tau_forall)
margin_i           = claim_i - counter_envelope_i
value              = tau_exists * logsumexp_i(margin_i / tau_exists)
```

As `tau_forall, tau_exists -> 0`, this approaches:

```
max_i [ claim_i - max_j counter_{ij} ]
```

i.e. the nested adversarial quantifier:

```
exists own candidate m such that no reply r refutes m.
```

Two temperatures (`tau_forall` and `tau_exists`) are kept independent
so the smoothing on the two quantifier levels can be tuned
separately. Both default to 0.20 and should be annealed during
training (0.5 -> 0.15 per the source packet).

The primitive returns `value`, `margin`, `counter_envelope`, the
witness and counterwitness soft assignments, indices of the best
witness / counterwitness, and the witness entropy.

The architecture-level claim is additive:

```
final_logit(x) = i193_trunk(x) + gate(x) * delta(x)
```

with `delta(x), gate(x)` small MLPs over `[witness-pooled cand,
counter-pooled reply, trunk_pool, value, max_margin,
counter_envelope_max, witness_entropy]` and
`[trunk_pool, value, max_margin, witness_entropy]` respectively.

## Why this matters

Equal-eval positions are hard because material and static pressure are
ambiguous. A position is puzzle-like only if one line survives
counterplay. WCQ expresses this directly. Promotion / underpromotion
near-puzzles and mate-in-1 near-puzzles are also defined by one
surviving defensive resource — a setting where WCQ should outperform
both `max claim only` and `mean counter penalty` baselines.

## Falsifier

- Primitive-level: `max_claim_only` disables the counter branch
  entirely; `mean_counter_penalty` replaces the forall-soft with a
  mean per-row penalty; `random_counter_assign` permutes counter rows
  across candidates so the counterwitness scores no longer match the
  candidate they reference; `no_counter_branch` zeros the counter
  scores.
- Architecture-level: p005 must improve matched-recall near-puzzle FP
  at recall 0.80 by at least 5% over i193 and i011 on the promotion /
  underpromotion and mate-in-1 buckets, without regressing aggregate
  PR AUC by more than 0.005.

## Composition with other Codex reply primitives

WCQ is one of five candidate/reply primitives compiled from the
2026-05-12 Codex batch. The hybrid stack treats the gated WCQ delta
as orthogonal to PAFR / RSP / RCC deltas; the trunk and the four
deltas sum without disturbing each other.

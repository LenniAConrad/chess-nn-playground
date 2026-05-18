# Math Thesis

`Near Puzzle Rejection Specialist` -- i256.

The benchmark contract is binary puzzle classification with the source-1 class
held out as verified near-puzzle. Let `y in {0, 1}` denote the binary label and
let `s` denote the per-board score. The deployment metric is `near_FP_rate` at
fixed `puzzle_recall in {0.80, 0.85}`, not threshold-free PR-AUC.

## Decomposition Identity

Let the board admit a (deterministic, board-only) candidate set `C(b)` for the
side to move. For each candidate `m in C(b)`, define

```text
claim(m)         in R        positive-tactical-claim score for m
reply_escape(m)  in R        best-surviving-defensive-reply score for m
forcedness_gap(m) = claim(m) - reply_escape(m)
```

The aggregate forcedness is the masked log-sum-exp / soft-max of
`forcedness_gap(m)` over `m in C(b)`. The model emits

```text
raw_claim_logit  in R        positive-tactical-claim summary
veto_logit       in R        survival-of-defence summary
final_logit      = raw_claim_logit - softplus(veto_logit)
```

`softplus(z) = log(1 + exp(z)) >= 0`, so `veto` is monotone in suppressing
positives and cannot lift a negative claim. This is the central identity that
gives the architecture its rejection bias.

## Per-square Candidate Pool

The candidate set `C(b)` is the set of board squares with non-zero
side-to-move attacker pressure. Concretely the simple_18 piece planes plus the
side-to-move plane are converted (deterministically, in tensor space) into
attacker masks by the `DualStreamFeatureBuilder` reused from i193. The mask
`m(s) in {0, 1}` over the 64 squares is a function only of the simple_18
tensor.

Aggregation is masked softmax over `s`:

```text
w(s)   = softmax(forcedness_gap(s)) * m(s) (renormalised)
soft_max_gap = sum_s w(s) * forcedness_gap(s)
entropy      = - sum_s w(s) log w(s)
eff_count    = exp(entropy)
top1_minus_top2 = max_s gap(s) - second_max_s gap(s)
```

For numerical robustness, rows with no candidates fall back to a uniform
distribution; downstream pools clamp the divisor away from zero. The selected
candidate count `|C(b)|` is exported as a diagnostic.

## Loss Identity

For the current shared-trainer implementation the loss is just

```text
L = BCEWithLogits(final_logit, y_binary)
```

with `final_logit = raw_claim - softplus(veto)`. The research packet's full
loss `L = L_main + lambda_gap * L_gap_rank + lambda_veto * L_veto` requires a
trainer extension to inject per-batch pair-matched terms. Those terms are not
mathematically required to make `final_logit` well-defined, and adding them
would couple the architecture promotion to a trainer change the rest of the
queue does not need. The keep / drop rule for that extension is documented in
`ablations.md`.

## Defender Overload Margin

For each own-piece square `d` the model learns a scalar
`overload_score(d) = MLP([trunk(d), exchange(d)])`. The construction is
intended to approximate the obligation-vs-safe-budget margin from the research
packet:

```text
obligation_count(d) ~ attacker pressure on critical targets, king-zone
                     coverage, recapture duties, ray-blocker duties,
                     promotion-stop duties.
safe_budget(d)      ~ safe mobility + alternative defenders.
overload_margin(d)  = obligation_count(d) - safe_budget(d)
```

The deterministic exchange feature stack from i193 already exposes attacker /
defender intensity, value and king-zone planes that the per-square MLP can use
as proxies for those quantities. Aggregation is masked softmax over the
own-piece mask.

## King Escape Pressure

The pooled king pressure feature is

```text
trunk_at_enemy_king  = mean_{s : enemy_king(s) = 1} trunk(s)
trunk_at_enemy_zone  = mean_{s : enemy_zone(s) = 1} trunk(s)
king_pressure        = MLP([trunk_at_enemy_king, trunk_at_enemy_zone,
                            king_feature_mean, board_summary])
```

Both masks come from the deterministic king feature stack (planes 1 and 3 of
the i193 king features are `enemy_king` and `enemy_zone`).

## Candidate Concentration

`concentration_score` consumes only the four scalar gap statistics
`[soft_max_gap, top1_minus_top2, entropy, |C(b)| / 64]`. This keeps
concentration explicitly downstream of the forcedness gap so it cannot
substitute for forcing.

## Veto Composition

The veto inputs are intentionally chess-explained scalars only:

```text
veto_input = [reply_escape_mass,
              forcedness_gap_entropy,
              -top1_minus_top2,
              |C(b)| / 64,
              overload_score,
              king_pressure,
              concentration_score]
veto_logit = MLP(veto_input)
```

`reply_escape_mass` is the masked-softmax expectation of the reply head over
candidates. `-top1_minus_top2` enters with a sign so that a *smaller* top-2 gap
(i.e. less concentrated forcing) increases the veto.

## Falsifiers

- `trunk_only` matches or beats `none`: the specialist is not load-bearing.
- `no_reply_envelope` matches `none`: the reply-side modelling is decorative.
- `no_overload_head` does not worsen the promotion / equal-eval slices: the
  overload story is not load-bearing.
- `no_king_escape_head` does not worsen the `mate_in_1` slice: the
  king-escape story is not load-bearing.
- Final logit fails the rejection identity (`final >> raw_claim` in any
  batch): a refactor has broken the `softplus`-only-subtract guarantee.

If any falsifier trips, the responsible head is reduced or removed. The keep
condition is matched-recall near-puzzle-FP reduction versus the i193 parent on
the canonical tagged split, evaluated with thresholds chosen on validation.

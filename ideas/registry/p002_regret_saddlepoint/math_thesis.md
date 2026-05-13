# Math Thesis

Source: `ideas/research/primitives/codex_02_regret_saddlepoint.md`
(Regret Saddlepoint Primitive, RSP).

## Working thesis

For a position `x`, let `A(x) in R^{K x R}` be a learned candidate /
reply payoff table (payoff to side-to-move). The primitive solves the
entropy-regularized saddle problem:

```
max_p min_q  p^T A q + tau_p H(p) - tau_q H(q)
subject to   p in Delta_K, q in Delta_R
```

The stationary equations are reached by damped fixed-point iteration:

```
row_pay = A q
p_new   = softmax(row_pay / tau_p)
col_pay = p_new^T A
q_new   = softmax(-col_pay / tau_q)
p, q    = (1 - damp) * (p, q) + damp * (p_new, q_new)
```

The primitive returns:

```
value           = p^T A q
attacker_regret = max_i row_pay_i - value
defender_regret = value - min_j col_pay_j
exploitability  = attacker_regret + defender_regret
```

plus both equilibrium strategies and per-side entropies.

The architecture-level claim is additive:

```
final_logit(x) = i193_trunk(x) + gate(x) * delta(x)
```

where `delta(x), gate(x)` are small MLPs over the attacker / defender
strategy-weighted pools, the trunk pool feature, and the scalar
diagnostics `(value, attacker_regret, defender_regret,
attacker_entropy, defender_entropy)`.

## Why this matters

The per-class puzzle_binary benchmark identifies near-puzzle false
positives at matched recall as the central pressure point. A near-
puzzle can have one huge-looking forcing row in `A` but one defensive
column that refutes it; the defender concentrates probability on the
refuting column and `value` collapses. A real puzzle should either
have a robust row or a high saddle value even under the defender's
best response. The exploitability scalar exposes the gap directly:
high value + low exploitability is robust forcing evidence; high
claim + low value is a tempting near-puzzle.

## Falsifier

- Primitive-level: row-shuffle (`row_shuffle_payoff`) or column-shuffle
  (`col_shuffle_payoff`) the payoff table; the game structure
  disappears and the saddle reduces to noise. The `uniform_payoff`
  ablation collapses the table to a per-batch mean (no game structure
  at all).
- Architecture-level: p002 must improve matched-recall near-puzzle FP
  at recall 0.80 by at least 3% over i193 without regressing aggregate
  PR AUC by more than 0.005, and the row/column shuffle ablations
  must lose most of the lift.

## Composition with other Codex reply primitives

RSP is one of five candidate/reply primitives compiled from the
2026-05-12 Codex batch:

- p001 (PAFR): partial-order frontier over candidate utility table;
  ignores payoff magnitudes.
- p003 (RCC): information-theoretic capacity of the reply
  distribution; ignores payoff values.
- p005 (WCQ): nested adversarial quantifier (exists / forall) over
  claim and counter scores; RSP solves the regularized game over the
  full payoff matrix.

A future hybrid can sum the gated deltas from any combination of
{PAFR, RSP, RCC, WCQ} without disturbing the trunk.

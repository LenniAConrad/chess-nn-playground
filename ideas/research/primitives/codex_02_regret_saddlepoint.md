# Regret Saddlepoint Primitive

Author: Codex
Model: GPT-5 (Codex coding agent)
Date: 2026-05-12
Status: research packet

## One-Line Claim

`primitive_regret_saddlepoint` is a differentiable entropy-regularized zero-sum game reducer for attacker-candidate versus defender-reply payoff tables.

## Why This Primitive Is Worth Adding

The last two primitives cover two nearby but different bottlenecks:

- `primitive_witness_counterwitness_quantifier`: does one witness survive all local replies?
- `primitive_pareto_antichain_frontier`: is the candidate landscape a clean nondominated frontier or a broad tradeoff set?

This primitive handles the missing middle case: a model has a candidate-by-reply payoff table, and the right summary is not just "best row" or "widest frontier", but the local saddle value and the regrets of both sides.

For puzzle-vs-near-puzzle classification, the interesting question is often:

```text
If side-to-move commits to its best tactical candidate,
where does the opponent concentrate defensive probability,
and how exploitable is each side?
```

A near-puzzle can have one huge-looking forcing row but one defensive column that refutes it. A true puzzle should have either one robust row or a high saddle value even under the defender's best response. The saddlepoint value and regret diagnostics give a future model a compact tactical "game pressure" object instead of a pooled candidate average.

## Mathematical Signature

Input:

```text
A: Float[B, K, R]      payoff to side-to-move for candidate i against reply j
M_i: Bool[B, K]        valid candidate mask
M_j: Bool[B, R]        valid reply mask
tau_p, tau_q           entropy temperatures
T                      solver iterations or implicit-solver tolerance
```

Output:

```text
value: Float[B]
p: Float[B, K]                  attacker mixed candidate weights
q: Float[B, R]                  defender mixed reply weights
attacker_regret: Float[B]
defender_regret: Float[B]
row_payoffs: Float[B, K]
col_payoffs: Float[B, R]
exploitability: Float[B]
```

The primitive solves the entropy-regularized saddle problem:

```text
max_p min_q  p^T A q + tau_p H(p) - tau_q H(q)
subject to   p in Delta(valid candidates), q in Delta(valid replies)
```

The stationary equations are:

```text
p_i proportional to exp((A q)_i / tau_p)
q_j proportional to exp(-(p^T A)_j / tau_q)
```

Then:

```text
value = p^T A q

row_payoffs = A q
col_payoffs = p^T A

attacker_regret = max_i row_payoffs_i - value
defender_regret = value - min_j col_payoffs_j
exploitability = attacker_regret + defender_regret
```

As `tau_p, tau_q -> 0`, the layer approaches a max-min saddle reducer. At finite temperature, it is smoother and produces useful gradients for multiple candidate and reply tokens.

## Why This Is Not Just WCQ Or Attention

Attention normalizes a scalar score and averages values. It does not compute an adversarial best response and it does not expose regret.

WCQ computes a pure nested quantifier over per-candidate counterwitness scores:

```text
max_i [ claim_i - max_j counter_ij ]
```

RSP consumes the full payoff matrix `A_ij` and solves a local regularized game. That matters when candidates and replies form cyclic or tradeoff structure: candidate A beats reply 1 but fails to reply 2, candidate B does the reverse, and no one row dominates. PAFR can say the frontier is broad; RSP says what the adversarial value and exploitability are.

A prototype can be written as unrolled softmax fixed-point updates. The primitive claim is the fused ragged masked saddle solver with implicit backward or controlled unrolled backward, saved equilibrium strategies, and regret diagnostics. PyTorch has no `nn` operator whose native output is a differentiable zero-sum game value plus both players' strategies and exploitability.

## Why This Fits The Benchmark

The benchmark's central pressure point is near-puzzle false positives at matched puzzle recall. The strongest current evidence says:

- `i193_exchange_then_king_dual_stream` is a strong parent because it separates exchange and king evidence.
- `i011_vetoselect` and `i191_safe_reply_certificate_verifier` help because they prevent tempting positive evidence from being accepted too cheaply.
- Promotion/underpromotion and mate-in-1 near-puzzles often differ from true puzzles by one defensive reply, not by the presence or absence of tactical texture.

RSP is a direct way to turn candidate/reply payoffs into a compact pressure score:

```text
high value, low exploitability   -> robust forcing evidence
high claim, low saddle value     -> tempting but refuted near-puzzle
high exploitability              -> unstable tactical texture
```

It is a good future architecture component because it can drop into the candidate/reply compiler already proposed for the Forcing Reply Envelope Veto model.

## Tiny Sanity Test

I ran a small PyTorch prototype with three payoff matrices:

1. True-like robust row:

```text
[3.0,  3.0, 3.0]
[1.2,  1.5, 1.1]
[0.5,  0.6, 0.7]
```

2. Near-like tempting row with one refutation:

```text
[3.0, -2.2, 3.0]
[1.0,  1.0, 1.0]
[0.7,  0.8, 0.7]
```

3. Cyclic tradeoff table:

```text
[ 2.6, -1.0,  0.2]
[ 0.2,  2.6, -1.0]
[-1.0,  0.2,  2.6]
```

With a damped soft saddle iteration, the outputs were:

```text
value:
true-like       2.953
near-like       0.845
cyclic          0.600

attacker weights p:
true-like       [0.975, 0.021, 0.005]
near-like       [0.084, 0.579, 0.336]
cyclic          [0.333, 0.333, 0.333]

defender weights q:
true-like       [0.335, 0.330, 0.336]
near-like       [0.224, 0.551, 0.224]
cyclic          [0.333, 0.333, 0.333]
```

The important behavior is that the near-like row does not get credit for the two high payoffs in row 0. The defender concentrates on the refuting column, and the attacker shifts to less flashy but safer rows.

## Minimal Torch Reference

```python
def regret_saddlepoint(A, cand_mask, reply_mask, tau_p=0.45, tau_q=0.45, iters=64, damp=0.35):
    # A: [B, K, R], payoff to side-to-move.
    B, K, R = A.shape
    neg = -1e9
    pos = 1e9

    p = cand_mask.float() / cand_mask.float().sum(-1, keepdim=True).clamp_min(1)
    q = reply_mask.float() / reply_mask.float().sum(-1, keepdim=True).clamp_min(1)

    for _ in range(iters):
        row_pay = torch.einsum("bkr,br->bk", A, q)
        row_pay = row_pay.masked_fill(~cand_mask, neg)
        p_new = torch.softmax(row_pay / tau_p, dim=-1)

        col_pay = torch.einsum("bk,bkr->br", p_new, A)
        col_score = (-col_pay).masked_fill(~reply_mask, neg)
        q_new = torch.softmax(col_score / tau_q, dim=-1)

        p = (1 - damp) * p + damp * p_new
        q = (1 - damp) * q + damp * q_new

    value = torch.einsum("bk,bkr,br->b", p, A, q)
    row_pay = torch.einsum("bkr,br->bk", A, q).masked_fill(~cand_mask, neg)
    col_pay = torch.einsum("bk,bkr->br", p, A).masked_fill(~reply_mask, pos)

    attacker_regret = row_pay.max(-1).values - value
    defender_regret = value - col_pay.min(-1).values
    exploitability = attacker_regret + defender_regret
    return value, p, q, attacker_regret, defender_regret, exploitability
```

This reference is only a correctness sketch. It is not the production implementation because naive unrolled backward can become unstable near cyclic equilibria. The real primitive should use either:

1. implicit differentiation through the regularized saddle equations; or
2. stop-gradient solver iterations plus one stable surrogate backward pass through the final row/column payoffs.

## Future Architecture Shape

Recommended first model:

```text
i193-style exchange/king parent
  -> candidate compiler
  -> reply compiler
  -> payoff table A_ij
  -> primitive_regret_saddlepoint
  -> VetoSelect-style head using:
       parent_pool,
       saddle_value,
       exploitability,
       attacker entropy,
       defender entropy,
       best candidate index,
       best reply index
```

Candidate rows:

- checks;
- captures;
- promotion pushes/captures;
- moves opening a slider line;
- moves attacking queen/rook/king-zone squares;
- high exchange-soundness moves.

Reply columns:

- recaptures;
- king escapes;
- interpositions;
- promotion stops;
- target defenses;
- counterchecks;
- local quiet defensive resources around affected squares.

The payoff table should be learned from board and candidate/reply token features. No engine evaluations, PVs, mate scores, source labels, or verification metadata enter the primitive.

## Falsification Tests

Primitive-level tests:

1. Row and column permutation equivariance: permuting candidates or replies must permute `p` or `q` and leave `value` unchanged.
2. Refutation test: decreasing one payoff in the defender's best column for the current best row must lower `value`.
3. Dominated row insertion: adding a strictly dominated candidate row should barely change `value`.
4. Dominated column insertion: adding a reply column that is worse for the defender should barely change `value`.
5. Low-temperature agreement: small examples should approach brute-force pure `max_i min_j A_ij` when the pure saddle is unique.
6. Exploitability decreases as solver iterations increase on well-conditioned random games.

Architecture-level test:

1. Add RSP to the candidate/reply branch of the Forcing Reply Envelope Veto model.
2. Compare against:
   - WCQ on the same candidates and replies;
   - pure `max_i min_j A_ij`;
   - scalar max over candidate claim;
   - ordinary attention over candidate/reply tokens;
   - row/column-shuffled payoff table preserving payoff histogram.
3. Keep parent trunk, candidate compiler, reply compiler, parameter budget, and split fixed.

Success criteria:

- matched-recall near-puzzle FP at recall `0.80` improves by at least `3%` versus the same parent without RSP;
- equal-eval and hard/very-hard slices improve or remain flat;
- promotion/underpromotion near-FP does not regress versus WCQ;
- payoff-table row/column shuffling destroys most of the gain, proving the game structure matters.

## Complexity

For `B` boards, `K` candidates, `R` replies, and `T` solver iterations:

```text
Forward:  O(B * T * K * R)
Backward: O(B * T * K * R) for unrolled, or O(B * K * R) plus linear-solve iterations for implicit backward
Memory:   O(B * K * R) plus saved strategies
```

For a chess-local compiler with `K <= 48`, `R <= 24`, and `T <= 32`, this is practical on a single RTX 3070 if the payoff table is chunked.

Incremental update:

```text
One changed candidate row: O(T * R)
One changed reply column:  O(T * K)
Full recompute:            O(T * K * R)
```

This makes it plausible for later search-style make/unmake benchmarks if the candidate/reply table is cached.

## Novelty Risk

The broad concept of differentiable optimization layers and entropy-regularized games is not new. The honest claim should be narrow:

> RSP is a reusable neural primitive for ragged masked zero-sum candidate/reply payoff tables, returning saddle value, strategies, and exploitability as first-class neural outputs.

It should not be sold as the first differentiable game solver. The chess-specific novelty is the operator boundary and the benchmark-facing use: a compact adversarial reducer between local forcing candidates and local defensive replies, replacing scalar pooling in hard near-puzzle discrimination.

The main risk is numerical. Entropy temperatures that are too low can create unstable fixed-point gradients, especially for cyclic tables. The first implementation should keep temperatures moderate, log diagnostics for exploitability and entropy, and compare unrolled-backward, implicit-backward, and stop-solver-gradient variants before making any benchmark claim.

## Recommendation

This primitive is high potential because it reuses the same infrastructure needed for WCQ: candidate tokens, reply tokens, and a payoff table. Prototype RSP after the padded WCQ branch exists. If RSP cannot beat WCQ or pure `max_i min_j` on matched-recall near-puzzle FP, keep WCQ and drop the saddle solver. If it wins specifically on equal-eval and cyclic-defense near-puzzles, it becomes a strong architecture candidate.

# Reply Channel Capacity Primitive

Author: Codex
Model: GPT-5 (Codex coding agent)
Date: 2026-05-12
Status: research packet

## One-Line Claim

`primitive_reply_channel_capacity` is a differentiable channel-capacity reducer over candidate-move to defender-reply distributions, measuring forced-reply information collapse.

## Why This Primitive Is Worth Adding

The current best benchmark hints make reply structure unusually important:

- `i192_latent_reply_entropy_network` is near the top of matched-recall near-puzzle false-positive results.
- `i191_safe_reply_certificate_verifier` is strong on promotion and underpromotion near-puzzle rejection.
- The newly proposed WCQ and RSP primitives both require a candidate/reply table, so a reply-information primitive can share the same future architecture infrastructure.

The missing primitive is an information-theoretic reducer for the reply table. Entropy alone asks whether replies are concentrated. Channel capacity asks a sharper question:

```text
How much can the attacking candidate choice control or reveal the defender reply distribution?
```

For true forcing tactics, good candidate moves should induce sharp, distinguishable defender-reply distributions. For near-puzzles, the reply landscape is often diffuse: many replies remain viable, or the tempting candidate does not create a unique defensive bottleneck. RCC turns that distinction into a reusable operator.

This is more conservative than the saddlepoint primitive. It is closer to a diagnostic information layer, but it has a clear path into a future architecture and directly targets the benchmark's hard-negative row.

## Mathematical Signature

Input:

```text
L: Float[B, K, R]      reply logits for reply j under candidate i
M_i: Bool[B, K]        valid candidate mask
M_j: Bool[B, R]        valid reply mask
T                      Blahut-Arimoto-style iterations or implicit tolerance
tau                    reply softmax temperature
```

Convert logits to a masked channel:

```text
P_ij = P(reply=j | candidate=i) = softmax_j(L_ij / tau)
```

Then solve for the capacity-achieving candidate prior:

```text
q* = argmax_q I_q(I; J)
subject to q in Delta(valid candidates)
```

where:

```text
r_j        = sum_i q_i P_ij
I_q(I; J) = sum_i q_i sum_j P_ij log(P_ij / r_j)
```

The primitive returns:

```text
capacity_nats:       Float[B]
q_star:              Float[B, K]
reply_marginal:      Float[B, R]
conditional_entropy: Float[B]
output_entropy:      Float[B]
row_entropy:         Float[B, K]
capacity_bits:       Float[B]
capacity_gap:        output_entropy - conditional_entropy
```

The capacity-achieving prior `q_star` is useful as candidate weights. `reply_marginal` is useful as the model's implied defensive bottleneck.

## Why This Is Not Just Reply Entropy

Reply entropy alone can be fooled by one sharp decoy candidate or by a broad reply distribution that is still highly candidate-dependent.

RCC uses the whole candidate-to-reply channel:

```text
candidate i -> distribution over replies j
```

It distinguishes:

- all candidates induce the same reply distribution: low capacity;
- each strong candidate induces a different narrow reply distribution: high capacity;
- one decoy candidate is sharp but the rest are broad: middling capacity, high conditional entropy.

That gives a richer model signal than `min_i H(reply | candidate=i)` or `mean_i H(reply | candidate=i)`.

It is also not WCQ or RSP:

- WCQ asks whether one witness survives all replies.
- RSP solves a zero-sum payoff game.
- RCC ignores payoff magnitudes and measures information structure in the reply distributions.

## Prior-Art Honesty

This should not be claimed as the first differentiable information solver. Blahut-Arimoto is classical channel-capacity machinery, and differentiable optimization layers such as OptNet establish the broader idea of embedding solvers in neural networks.

The narrow primitive claim is:

> RCC is a reusable ragged masked neural operator for candidate/reply channels, returning channel capacity, capacity-achieving candidate weights, reply marginals, and forced-reply entropy diagnostics as first-class outputs.

If a recent paper already exposes the same ragged channel-capacity operator with these outputs, downgrade this packet to "underexplored primitive for chess reply modeling."

References worth checking before publication:

- Blahut, "Computation of channel capacity and rate-distortion functions", IEEE Transactions on Information Theory, 1972.
- Arimoto, "An algorithm for computing the capacity of arbitrary discrete memoryless channels", IEEE Transactions on Information Theory, 1972.
- Amos and Kolter, "OptNet: Differentiable Optimization as a Layer in Neural Networks", ICML 2017.

## Tiny Sanity Test

I ran a small PyTorch Blahut-Arimoto-style prototype with three reply channels.

Case 1: distinct forced rows:

```text
[5.0, 0.0, 0.0, 0.0]
[0.0, 5.0, 0.0, 0.0]
[0.0, 0.0, 5.0, 0.0]
```

Case 2: diffuse near-like rows:

```text
[1.0, 0.9, 0.8, 0.7]
[0.8, 0.9, 1.0, 0.7]
[0.9, 0.8, 0.7, 1.0]
```

Case 3: one sharp decoy row plus broad rows:

```text
[5.0, 0.0, 0.0, 0.0]
[1.0, 1.0, 1.0, 1.0]
[1.0, 1.0, 1.0, 1.0]
```

Outputs:

```text
capacity_nats:
forced-distinct  1.012
diffuse-near     0.006
sharp-decoy      0.316

capacity_bits:
forced-distinct  1.460
diffuse-near     0.008
sharp-decoy      0.456

conditional_entropy:
forced-distinct  0.119
diffuse-near     1.380
sharp-decoy      0.869

score = capacity - 0.35 * conditional_entropy:
forced-distinct   0.970
diffuse-near     -0.477
sharp-decoy       0.012
```

This is the intended behavior. The primitive rewards a structured forced-reply channel, rejects diffuse reply landscapes, and does not over-reward a single sharp decoy candidate when the broader candidate channel remains noisy.

## Minimal Torch Reference

```python
def reply_channel_capacity(logits, cand_mask, reply_mask, iters=64, tau=1.0):
    # logits: [B, K, R]
    B, K, R = logits.shape
    neg = -1e9
    eps = 1e-8

    row_logits = logits / tau
    row_logits = row_logits.masked_fill(~reply_mask[:, None, :], neg)
    P = torch.softmax(row_logits, dim=-1)
    P = P * reply_mask[:, None, :].float()

    q = cand_mask.float() / cand_mask.float().sum(-1, keepdim=True).clamp_min(1)

    for _ in range(iters):
        r = torch.einsum("bk,bkr->br", q, P).clamp_min(eps)
        d = (P.clamp_min(eps) * (P.clamp_min(eps).log() - r[:, None, :].log())).sum(-1)
        d = d.masked_fill(~cand_mask, neg)
        q = torch.softmax(d, dim=-1)

    r = torch.einsum("bk,bkr->br", q, P).clamp_min(eps)
    d = (P.clamp_min(eps) * (P.clamp_min(eps).log() - r[:, None, :].log())).sum(-1)
    d = d.masked_fill(~cand_mask, 0.0)
    capacity = (q * d).sum(-1)

    row_entropy = -(P.clamp_min(eps) * P.clamp_min(eps).log()).sum(-1)
    conditional_entropy = (q * row_entropy).sum(-1)
    output_entropy = -(r * r.log()).sum(-1)

    return capacity, q, r, conditional_entropy, output_entropy, row_entropy
```

Production should avoid a long autograd tape through all iterations. Use either implicit differentiation through the fixed point or a stop-solver-gradient variant with a surrogate backward through the final divergence terms.

## Future Architecture Path

Recommended first use:

```text
i193-style exchange/king parent
  -> candidate compiler
  -> reply compiler
  -> reply logits L_ij
  -> primitive_reply_channel_capacity
  -> head receives:
       parent_pool,
       capacity_nats,
       conditional_entropy,
       output_entropy,
       capacity-achieving q_star summary,
       reply_marginal summary
```

This also fits naturally beside WCQ and RSP:

```text
WCQ:  witness survival
RSP:  payoff-game value and exploitability
RCC:  forced-reply information capacity
PAFR: candidate tradeoff frontier width
```

Together these give future architectures several orthogonal views of the same candidate/reply system.

## Falsification Tests

Primitive-level tests:

1. Candidate and reply permutation equivariance.
2. Identical reply rows produce capacity near zero.
3. Distinct one-hot reply rows produce capacity near `log(min(K, R))`.
4. Adding a duplicate candidate row should not increase capacity materially.
5. Masked padded implementation must match ragged implementation to `1e-5`.
6. Capacity should be nonnegative and no larger than `log(R_valid)` up to numerical tolerance.

Architecture-level test:

1. Add RCC to the same candidate/reply branch used for WCQ.
2. Compare against:
   - reply entropy only;
   - min row entropy;
   - mean row entropy;
   - RSP saddle value only;
   - scalar candidate max;
   - row-shuffled reply channels preserving per-row entropy.
3. Keep the parent trunk, candidate/reply compiler, and parameter budget fixed.

Success criteria:

- matched-recall near-puzzle FP at recall `0.80` improves by at least `2%` versus reply-entropy-only on the same parent;
- `i192`-style entropy benefits are retained, but row-shuffled reply channels destroy most of the extra RCC gain;
- promotion/underpromotion near-FP does not regress versus WCQ or safe-reply baselines;
- capacity diagnostics separate true puzzles from near-puzzles more cleanly than conditional entropy alone.

## Complexity

For `B` boards, `K` candidates, `R` replies, and `T` capacity iterations:

```text
Forward:  O(B * T * K * R)
Backward: O(B * T * K * R) for unrolled, or O(B * K * R) plus fixed-point backward
Memory:   O(B * K * R)
```

For chess-local `K <= 48`, `R <= 24`, and `T <= 32`, the padded prototype is feasible on a single RTX 3070.

Incremental update:

```text
Changed candidate row: O(T * R)
Changed reply column:  O(T * K)
Full recompute:        O(T * K * R)
```

The operator is not as cheap as WCQ, but its diagnostics are richer and it can share candidate/reply tensors with other primitives.

## Risks

The main risk is that channel capacity captures generic branching diversity rather than chess forcing. That is why entropy-only, row-shuffled, duplicate-row, and reply-count controls are mandatory.

The second risk is optimization overhead. If unrolled Blahut-Arimoto is slow or unstable, use a small fixed iteration count with solver-state stop-gradient first. This is acceptable for the scout because the goal is to test whether capacity diagnostics carry signal, not to publish the perfect differentiable solver.

The third risk is duplication with `i192_latent_reply_entropy_network`. This packet is only worthwhile if the capacity-achieving prior and mutual-information channel structure beat entropy-only controls. If they do not, keep the simpler entropy architecture.

## Recommendation

Prototype RCC after the candidate/reply compiler exists. It is conservative, benchmark-aligned, and likely easier to stabilize than the saddlepoint solver. The best first architecture is not a standalone RCC model; it is an i193-style parent with a small candidate/reply branch that exports WCQ, RSP, and RCC diagnostics to a VetoSelect head.

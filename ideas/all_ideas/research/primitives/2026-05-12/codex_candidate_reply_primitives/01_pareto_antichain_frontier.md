# Pareto Antichain Frontier Primitive

Author: Codex
Model: GPT-5 (Codex coding agent)
Date: 2026-05-12
Status: research packet

## One-Line Claim

`primitive_pareto_antichain_frontier` is a differentiable partial-order reducer that extracts the nondominated tactical candidate frontier instead of forcing all evidence through one scalar score too early.

## Why This Primitive Is Worth Adding

The current primitive batch already covers delta accumulators, legal graph routing, ray scans, blocker semirings, exchange reducers, symmetry operators, and the new witness-counterwitness quantifier. The missing thing is not another way to aggregate one score. It is an operator for the case where each tactical candidate has several non-commensurate utility axes:

- forcing claim;
- king exposure;
- exchange soundness;
- reply safety;
- promotion pressure;
- line-opening value;
- defender overload;
- tempo cost.

True puzzles often have a candidate that is strong across most axes, so the candidate set collapses to a small nondominated frontier. Near-puzzles often have a tempting candidate that is excellent on one axis but bad on another. A scalar max, ordinary attention, or learned weighted sum can overfire because it linearizes the tradeoff before seeing the set geometry.

This primitive keeps the partial order intact. It asks:

```text
Which candidate moves are not dominated by any other candidate across the learned tactical axes?
```

Then it exposes frontier width, frontier entropy, and nondominated summaries to the model. That is a different error detector from `primitive_witness_counterwitness_quantifier`: WCQ tests whether one witness survives local replies; PAFR tests whether the whole candidate landscape has a clean tactical frontier or a messy unresolved tradeoff.

## Mathematical Signature

Input:

```text
U: Float[B, K, C]       candidate utility channels, larger is better
M: Bool[B, K]           valid candidate mask
V: Float[B, K, D]       optional candidate value vectors to summarize
eps: Float[C]           strict dominance margin
tau_dim, tau_set        temperatures
```

For candidates `i` and `j`, define soft channelwise dominance:

```text
s_ijc = (U_i,c - U_j,c - eps_c) / tau_dim
p_ij  = product_c sigmoid(s_ijc)
```

`p_ij` is the soft probability that candidate `i` dominates candidate `j` on every channel. Mask invalid candidates and set `p_jj = 0`.

The probability that `j` is nondominated is:

```text
log_pi_j = sum_{i != j} log(1 - p_ij)
pi_j     = exp(log_pi_j)
```

With optional scalar quality `q_j = a^T U_j`, frontier attention is:

```text
alpha_j = softmax_j((log_pi_j + beta * q_j) / tau_set)
```

Outputs:

```text
frontier_summary = sum_j alpha_j V_j
frontier_width   = sum_j pi_j
frontier_entropy = -sum_j alpha_j log(alpha_j)
dominance_matrix = p_ij
nondominated_prob = pi_j
frontier_weights = alpha_j
```

As `tau_dim -> 0`, this approaches the exact Pareto antichain indicator: `pi_j = 1` if no valid candidate strictly dominates `j`, otherwise `0`.

## Why This Is Not Just Existing Attention Or Top-K

Attention, sparsemax, and top-k all require a scalar score per item. They choose items after the model has already collapsed the utility axes into one ordering.

This primitive computes a partial order before scalarization. The critical internal object is the pairwise dominance matrix over all candidate pairs and all utility channels. A differentiable sort imposes a total order; PAFR computes a soft antichain of a product order.

A padded prototype can be written with broadcasted PyTorch ops, just as a prototype attention layer can be written with matmul, softmax, and matmul. The primitive claim is the fused log-domain dominance/antichain reducer with saved pairwise dominator adjoints. The useful backward path is not "winner gets gradient" but "frontier candidates, dominated candidates, and their closest dominators share gradient according to the partial-order slack."

## Chess Motivation

The repo's strongest recent evidence points at hard-negative discrimination, not easy aggregate accuracy:

- `i193_exchange_then_king_dual_stream` is the best broad parent, implying exchange and king evidence should stay separated.
- `i011_vetoselect` and `i191_safe_reply_certificate_verifier` improve key near-puzzle rejection slices by separating accepted evidence from rejected evidence.
- Equal-eval and hard/very-hard rows remain the stable weak slices, where scalar material/eval shortcuts are least useful.

PAFR is built for positions where no single scalar tactical score is trustworthy. A real puzzle should often show a narrow nondominated frontier: the winning candidate is good on claim, safety, and target pressure. A near-puzzle may show a wide frontier: one move has claim, another has safety, another has material, and no candidate cleanly dominates.

## Tiny Sanity Test

I ran a toy PyTorch check with `C = 3` utility channels.

True-like row:

```text
[4.0, 0.95, 0.90]
[3.0, 0.45, 0.40]
[2.2, 0.60, 0.55]
[1.5, 0.70, 0.30]
```

Ambiguous near-like row:

```text
[4.0, 0.12, 0.30]
[3.0, 0.82, 0.72]
[2.2, 0.90, 0.80]
[1.5, 0.70, 0.60]
```

The output was:

```text
not_dominated_prob
true-like:      [1.000, 0.006, 0.035, 0.050]
ambiguous-like: [1.000, 1.000, 0.959, 0.087]

frontier_weights
true-like:      [1.000, 0.000, 0.000, 0.000]
ambiguous-like: [0.367, 0.389, 0.244, 0.000]

soft_frontier_size
true-like:      1.091
ambiguous-like: 3.046

frontier_entropy
true-like:      0.000
ambiguous-like: 1.080
```

This is the intended behavior: the primitive does not merely pick the biggest claim. It reports whether the tactical candidate set has a single clean nondominated explanation or a broad unresolved tradeoff frontier.

## Minimal Torch Reference

```python
def pareto_antichain_frontier(U, mask=None, V=None, tau_dim=0.08, tau_set=0.25, eps=0.03, beta=0.35):
    # U: [B, K, C], larger is better.
    B, K, C = U.shape
    if V is None:
        V = U
    if mask is None:
        mask = torch.ones(B, K, dtype=torch.bool, device=U.device)

    diff = U[:, :, None, :] - U[:, None, :, :] - eps
    p_dom = torch.sigmoid(diff / tau_dim).prod(dim=-1)

    eye = torch.eye(K, dtype=torch.bool, device=U.device)[None]
    valid_pair = mask[:, :, None] & mask[:, None, :] & ~eye
    p_dom = torch.where(valid_pair, p_dom, p_dom.new_zeros(()))

    log_pi = torch.log1p(-p_dom.clamp(max=1 - 1e-6)).sum(dim=1)
    log_pi = torch.where(mask, log_pi, log_pi.new_full((), -1e9))
    pi = log_pi.exp()

    q = U.mean(dim=-1)
    alpha = torch.softmax((log_pi + beta * q) / tau_set, dim=-1)
    summary = torch.einsum("bk,bkd->bd", alpha, V)
    entropy = -(alpha * (alpha + 1e-9).log()).sum(dim=-1)
    width = (pi * mask.float()).sum(dim=-1)
    return summary, width, entropy, pi, alpha, p_dom
```

This reference is intentionally simple. A real primitive should keep the dominance computation in log space and fuse the pairwise reduction to avoid materializing unnecessary intermediates for large `K`.

## How To Use It In The Current Research Stack

Recommended first instantiation:

```text
Candidate set K:
  side-to-move forcing candidates or high-salience square/candidate tokens

Utility channels C:
  claim score
  exchange soundness
  king-zone pressure
  reply-survival score
  promotion/lane score
  defender-overload score
```

The utility channels should be learned projections from the model's internal token states. Deterministic rule features may compile the candidate set, but the primitive should not read engine scores, PVs, mate distance, verification metadata, or source labels.

PAFR can sit before a VetoSelect-style head:

```text
accepted_evidence = frontier_summary
ambiguity_signal  = [frontier_width, frontier_entropy]
```

For puzzle-vs-near-puzzle classification, high claim plus high frontier entropy is suspicious: it means the model sees tactical texture but cannot find one clean nondominated line.

## Falsification Tests

Primitive-level tests:

1. Candidate permutation invariance: permuting candidate rows must permute `pi` and `alpha` but leave `summary`, `width`, and `entropy` unchanged.
2. Dominated insertion stability: adding a clearly dominated candidate should barely change `summary`.
3. Tradeoff insertion sensitivity: adding a nondominated tradeoff candidate should increase `frontier_width` and usually increase `frontier_entropy`.
4. Hard-limit check: as `tau_dim` decreases, `pi` should converge to exact nondominated membership on small brute-force examples.
5. `gradcheck` should pass in double precision away from exact dominance ties.

Architecture-level scout test:

1. Add PAFR to the candidate/reply head of `Forcing Reply Envelope Veto Network`, or to a small i193-style exchange/king parent with a candidate utility table.
2. Compare against:
   - scalar max over claim;
   - ordinary attention over candidates;
   - sparsemax/top-k over scalar candidate scores;
   - PAFR with utility channels shuffled across candidates;
   - PAFR with only one utility channel.
3. Keep the candidate compiler, parameter budget, training split, and thresholding protocol fixed.

Success criteria:

- matched-recall near-puzzle false positives at recall `0.80` improve by at least `3%` over the same parent without PAFR;
- equal-eval PR AUC improves over the parent, because scalar shortcuts should be less dominant;
- promotion/underpromotion near-FP does not regress versus the WCQ/reply-envelope variant;
- PAFR beats the one-channel ablation, proving the partial-order structure matters beyond another pooling layer.

## Complexity

For `B` boards, `K` candidates, and `C` utility channels:

```text
Forward:  O(B * K^2 * C)
Backward: O(B * K^2 * C)
Memory:   O(B * K^2) if the dominance matrix is saved
```

For chess candidate sets with `K <= 64` and `C <= 8`, this is cheap compared with full-board attention over all square pairs and usually small compared with a residual tower.

Incremental update:

```text
One changed candidate row: O(K * C)
Delta candidate insertion/removal: O(DeltaK * K * C)
Worst case full recompute: O(K^2 * C)
```

That bounded-change update matters for search settings where sibling nodes share most candidate structure.

## Novelty Risk

The strongest reviewer objection is that the padded prototype decomposes into broadcasted subtraction, sigmoid, product, log, and softmax. That is true for a prototype. The primitive-worthy claim is narrower:

> PAFR is a reusable differentiable partial-order antichain reducer with a distinctive pairwise dominance backward path and log-domain fused implementation.

It is not a new activation, not a new loss, and not a new attention mask. It is closest to differentiable sorting, skyline/Pareto database operators, and multi-objective optimization layers. If a recent differentiable Pareto-front layer already exists with the same operator boundary and backward semantics, this packet should be downgraded from "new primitive" to "underexplored primitive for chess tactical candidates."

## Recommendation

This is worth prototyping only after the candidate utility table exists for WCQ or the reply-envelope model. The first padded version is simple and should take less than a day. Do not build a fused kernel until the one-channel, shuffled-channel, and scalar-attention ablations prove that partial-order frontier geometry improves near-puzzle rejection.

# Tail Copula Concordance Primitive

Author: Codex
Model: GPT-5 (Codex coding agent)
Date: 2026-05-12
Status: research packet

## One-Line Claim

`primitive_tail_copula_concordance` is a differentiable rank-copula reducer that measures whether multiple tactical evidence fields become extreme on the same squares or candidates.

## Why This Primitive Is Worth Adding

The project already has a `rank_quantile_evidence_field_network` (`i095`) that tests whether extreme scalar evidence fields help more than mean pooling. That is useful, but it only sees marginal tails. It cannot distinguish these two cases:

```text
Case A: exchange, king-pressure, and reply-danger all spike on the same square.
Case B: each channel has the same top quantiles, but their spikes occur on different squares.
```

For puzzle-vs-near-puzzle classification, that distinction is likely important. A real tactic often needs multiple independent evidence streams to become extreme at the same tactical site: a king line, an exchange swing, an overloaded defender, a promotion lane, or a reply bottleneck. A near-puzzle can have the same marginal amount of tactical texture spread across unrelated sites.

The primitive therefore measures upper-tail dependence between evidence channels after converting each channel to soft ranks. It is conservative: it does not generate moves, solve a game, or enumerate proof trees. It is also different from ordinary correlation because it ignores the bulk of the board and asks whether the extremes co-activate.

## Mathematical Signature

Input:

```text
X: Float[B, N, C]      evidence values over N sites and C channels
M: Bool[B, N]          valid site mask
q: Float              upper-tail threshold, e.g. 0.75 or 0.90
tau_rank              soft-rank temperature
tau_tail              tail-membership temperature
```

For each channel `c`, compute a soft empirical CDF/rank:

```text
u_bnc = soft_rank(X_bnc over n | M_b) / N_valid
```

Then compute soft upper-tail membership:

```text
m_bnc = sigmoid((u_bnc - q) / tau_tail)
```

The directional upper-tail dependence estimate is:

```text
lambda_{c -> d}
  = sum_n m_bnc * m_bnd / (eps + sum_n m_bnc)
```

Use a symmetric concordance matrix:

```text
Lambda_cd = sqrt(lambda_{c -> d} * lambda_{d -> c})
```

Outputs:

```text
tail_matrix:       Float[B, C, C]
tail_mean:         Float[B]
tail_max:          Float[B]
tail_entropy:      Float[B]
tail_eigenvalues:  Float[B, C] optional
tail_site_mass:    Float[B, N]
channel_tail_mass: Float[B, C]
```

`tail_site_mass` can be the per-site average or max of pairwise co-tail products, giving the later architecture a soft tactical hotspot map.

## Why This Is Not Rank-Quantile Pooling

Rank-quantile pooling sorts each channel independently. It knows the shape of each marginal evidence distribution but not whether the upper tails line up.

TCC first transforms each channel to marginal ranks, then computes dependence between the ranked tails. That makes it invariant to monotone rescaling of individual evidence channels while preserving cross-channel co-extremity. Two boards can have identical per-channel sorted values and identical quantiles but very different TCC matrices.

It is also not ordinary correlation:

- Pearson correlation is dominated by bulk behavior and scale.
- Spearman/Kendall-style rank correlation measures whole-distribution concordance.
- TCC focuses on the upper tail only, which matches the benchmark pressure: sparse tactical evidence on a few critical sites.

## Prior-Art Honesty

The broad statistical objects are not new. Copulas separate marginal distributions from dependence structure, tail-dependence coefficients measure joint extreme behavior, and differentiable ranking/sorting has several known relaxations.

The narrow primitive claim is:

> TCC is a reusable neural operator that combines differentiable ranks, soft tail membership, and tail-dependence concordance into a first-class layer over evidence fields or candidate tables.

It should not be claimed as the first copula method or the first differentiable rank operator. Its novelty for this project is the operator boundary and the benchmark target: measuring whether independent chess evidence streams become extreme together in hard near-puzzle cases.

References worth checking before publication:

- Sklar-style copula theory and tail-dependence coefficients for separating marginals from dependence.
- Cuturi, Teboul, and Vert, "Differentiable Ranking and Sorting using Optimal Transport", NeurIPS 2019.
- Recent differentiable sorting and ranking operators such as NeuralSort, SoftSort, and differentiable sorting networks.

## Tiny Sanity Test

I ran a small PyTorch prototype with `N = 8` sites and `C = 3` channels.

Case 1: all channels spike on the same top sites.

Case 2: each channel has the same marginal ranked values, but the spikes are shifted to different sites.

Case 3: diffuse mildly varying channels.

The output was:

```text
mean_upper_tail_dependence:
same-site extreme       0.834
disjoint-site extreme   0.025
diffuse                 0.207
```

The tail-dependence matrices were:

```text
same-site extreme:
[[0.834, 0.834, 0.834],
 [0.834, 0.834, 0.834],
 [0.834, 0.834, 0.834]]

disjoint-site extreme:
[[0.834, 0.034, 0.005],
 [0.034, 0.834, 0.034],
 [0.005, 0.034, 0.834]]

diffuse:
[[0.251, 0.160, 0.302],
 [0.160, 0.259, 0.158],
 [0.302, 0.158, 0.375]]
```

Importantly, the same-site and disjoint-site cases can have identical per-channel sorted values. Marginal quantiles alone cannot separate them. TCC can.

## Minimal Torch Reference

```python
def soft_rank_uniform(x, mask=None, tau_rank=0.35):
    # x: [B, N, C], higher value -> higher rank in (0, 1].
    B, N, C = x.shape
    if mask is None:
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)

    diff = x[:, :, None, :] - x[:, None, :, :]
    pair_mask = mask[:, :, None, None] & mask[:, None, :, None]
    soft_less_equal = torch.sigmoid(diff / tau_rank)
    soft_less_equal = torch.where(pair_mask, soft_less_equal, soft_less_equal.new_zeros(()))
    valid_n = mask.float().sum(dim=1).clamp_min(1.0)
    rank = soft_less_equal.sum(dim=2) / valid_n[:, None, None]
    return rank


def tail_copula_concordance(x, mask=None, q=0.75, tau_rank=0.35, tau_tail=0.06):
    # x: [B, N, C]
    if mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

    u = soft_rank_uniform(x, mask, tau_rank=tau_rank)
    m = torch.sigmoid((u - q) / tau_tail) * mask[:, :, None].float()

    numer = torch.einsum("bnc,bnd->bcd", m, m)
    denom = m.sum(dim=1).clamp_min(1e-6)
    directional = numer / denom[:, :, None]
    tail_matrix = torch.sqrt((directional * directional.transpose(1, 2)).clamp_min(0))

    C = x.size(-1)
    eye = torch.eye(C, dtype=torch.bool, device=x.device)[None]
    offdiag = tail_matrix.masked_select(~eye).view(x.size(0), C, C - 1)
    tail_mean = offdiag.mean(dim=(1, 2))
    tail_max = offdiag.max(dim=2).values.max(dim=1).values
    tail_site_mass = (m[:, :, :, None] * m[:, :, None, :]).mean(dim=(2, 3))
    channel_tail_mass = m.sum(dim=1)

    return tail_matrix, tail_mean, tail_max, tail_site_mass, channel_tail_mass
```

This reference is intentionally simple and quadratic in sites for the rank step. For chess `N = 64`, that is acceptable for a prototype. A future fused primitive could use differentiable sorting/ranking kernels to reduce cost.

## Future Architecture Path

Recommended first use:

```text
i193-style exchange/king parent
  -> produce evidence fields over squares or candidates:
       exchange swing
       king-zone pressure
       line-opening pressure
       defender overload
       reply danger
       promotion lane pressure
  -> primitive_tail_copula_concordance
  -> VetoSelect-style head receives:
       parent_pool,
       tail_mean,
       tail_max,
       tail eigen summaries,
       tail hotspot map pooled around kings / promotion lanes
```

This can also be used on candidate tables:

```text
X[B, K, C] = candidate utility channels
```

Then TCC measures whether the same candidates occupy the upper tails across multiple utility axes. That complements `primitive_pareto_antichain_frontier`: PAFR asks about nondominated candidates; TCC asks whether extreme evidence aligns on the same candidates.

## Falsification Tests

Primitive-level tests:

1. Monotone invariance: applying a strictly increasing transform to one channel should barely change ranks and tail dependence.
2. Marginal-sort control: two inputs with identical per-channel sorted values but different site alignment should produce different `tail_matrix`.
3. Channel permutation equivariance: permuting channels permutes `tail_matrix`.
4. Site permutation invariance: jointly permuting sites leaves `tail_matrix` unchanged.
5. Independent random channels should produce low off-diagonal tail dependence.
6. Perfectly co-monotone channels should produce high off-diagonal tail dependence.

Architecture-level test:

1. Add TCC after an i193-style exchange/king evidence-field trunk.
2. Compare against:
   - rank-quantile pooling only (`i095`-style);
   - Pearson/Spearman-style whole-field correlation;
   - mean/max pooling;
   - square-shuffled TCC preserving per-channel values;
   - channel-shuffled TCC preserving each field's marginal distribution.
3. Keep parent trunk, parameter budget, input encoding, and split fixed.

Success criteria:

- matched-recall near-puzzle FP at recall `0.80` improves by at least `2%` over rank-quantile pooling only;
- gains survive equal-eval and hard/very-hard slices;
- square-shuffled TCC destroys most of the gain, proving co-location of tail evidence matters;
- promotion/underpromotion does not regress versus the reply/capacity primitives.

## Complexity

For `B` boards, `N` sites, and `C` evidence channels:

```text
Soft-rank prototype: O(B * N^2 * C)
Tail matrix:         O(B * N * C^2)
Backward:            same order
Memory:              O(B * N^2 * C) if pairwise rank comparisons are saved
```

For chess squares, `N = 64`, so the prototype is feasible. For candidate tables, `K <= 64` is also feasible. A production primitive should use differentiable sort/rank kernels or bucketed soft CDF approximations.

Incremental update:

```text
One changed site: O(N * C + N * C^2) with cached ranks/tail masses
Full recompute:   O(N^2 * C + N * C^2)
```

## Risks

The strongest duplicate risk is `i095_rank_quantile_evidence_field_network`. TCC is only worth keeping if the square-shuffle and marginal-sort controls show that cross-channel tail alignment adds information beyond marginal quantiles.

The second risk is that TCC may reward generic co-activation artifacts such as material imbalance or phase. The benchmark must report material/eval buckets and use a channel-shuffle or square-shuffle control.

The third risk is temperature sensitivity. If `tau_rank` is too high, the operator becomes smooth bulk correlation; if too low, gradients become sparse and noisy. The first sweep should test `q in {0.70, 0.80, 0.90}` and `tau_rank in {0.2, 0.35, 0.5}`.

## Recommendation

Prototype TCC as a small readout on top of the i193 exchange/king evidence fields, not as a standalone architecture. It is a conservative primitive with a clean falsifier: if marginal rank-quantiles match it, drop TCC. If it wins specifically on hard near-puzzles while square-shuffled tails fail, it becomes a strong companion to WCQ/RSP/RCC for future candidate/reply models.

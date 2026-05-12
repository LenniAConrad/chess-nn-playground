# 01 — Signed Piece-Existence Hessian Operator (DHPE)

**Slug:** `primitive_signed_hessian_resonance`
**Status:** proposed
**Author:** Claude
**Model:** Claude Opus 4.7
**Architecture extension:** i245 Pair-Resonance Hessian Network

## One-line claim

A forward-pass primitive that computes the **discrete second-order mixed
forward-difference** of a learned scoring function with respect to pairwise
piece-existence indicators, exposing the **signed** super- or sub-additivity
of every piece pair as a chess-tactical resonance tensor.

## Mathematical signature

For a position with active piece set `P`, a learned scoring function
`phi_theta : 2^P -> R^d` (subsets of pieces -> feature vector), define

    H_ij(x) = phi(P) - phi(P \ {i}) - phi(P \ {j}) + phi(P \ {i, j})

This is the second discrete forward-difference of `phi` with respect to the
Boolean piece-presence indicators `s_i, s_j in {0, 1}`. The primitive's
output is the matrix `H in R^{n x n}` (symmetric, zero diagonal) — or its
top-k subsampling.

The sign of `H_ij` carries chess meaning:

| sign(H_ij) | algebra | chess meaning |
|---|---|---|
| `H_ij > 0` (super-additive) | `phi(P) + phi(P\{i,j}) > phi(P\{i}) + phi(P\{j})` | pieces i and j *jointly* create value — both must be present (a tactic) |
| `H_ij < 0` (sub-additive)   | inequality reverses | pieces i and j *substitute* — removing either makes the other more critical (defender / blocker relationship) |
| `H_ij ~ 0`                    | additive | independent pieces, no interaction |

## Why this is genuinely new

Not in the 12 known primitive families:

- Family 1 (delta accumulator): no incremental state; pure forward differences
- Family 2 (ray scan): no directional propagation
- Family 3 (legal-move graph): no edge structure on the graph; the piece interaction graph is *induced by data*
- Family 4 (group equivariance): no group action
- Family 5 (hyperedge): pair-wise but operates on *forward-difference quantity*, not contraction product
- Family 6 (tropical semiring): standard arithmetic, no max-plus
- Family 7 (DEQ): no fixed-point iteration
- Family 8 (SSM-on-topology): no scan
- Family 9 (reversible): no inverse
- Family 10 (bilinear-attack-defend): bilinear computes `u_i^T W v_j`; DHPE computes 4-point forward-difference combination
- Family 11 (hypernetwork): doesn't generate weights
- Family 12 (group-orbit norm): no orbit averaging

Closest existing in the i### registry:

| existing | operates on | order | output |
|---|---|---|---|
| i189 Counterfactual Defender Dropout | single-piece dropout | unary | 13-d sensitivity vector |
| i041 Centered Tempo-Odd Interventional | tempo + null-board | unary | scalar bottleneck |
| i211 Role-Counterfactual Necessity | role counterfactual | unary | necessity bottleneck |
| i025-i027 | move-level (not piece) | unary moves | move-landscape spectrum |
| i222 Schur Defender Elimination | algebraic block elimination | linear algebra | Schur features |
| **DHPE (this)** | **piece x piece pair** | **second-order, same-type** | **signed pair-Hessian tensor** |

The closest analogue in mainstream ML is integrated gradients or Shapley
approximations, but those operate on continuous gradients; DHPE operates via
*discrete forward differences*.

## Empirical evidence from prototypes

Two local prototype variants are checked in under `prototypes/`: a hand-evaluated
combinatorial scorer (`dhpe_prototype.py`) and a learned PyTorch scorer
(`dhpe_v2.py`). They are design checks only, not scout-scale benchmark runs.

**Hand-crafted scorers with planted interactions:**

| Scenario | L1(H) | entropy | max\|H\| | top pair | sign |
|---|---:|---:|---:|---|---|
| PIN ({attacker, pinned, target}) | 30 | 1.099 | 10 | (attacker, pinned) | **+10** |
| FORK ({forker, target1, target2}) | 30 | 1.099 | 10 | (forker, target1) | **+10** |
| NEUTRAL (additive) | 0 | 0 | 0 | (any) | 0 |
| NEAR-PUZZLE (tactic only if defender absent) | 30 | 1.099 | 10 | (attacker, **defender**) | **-10** |

**Critical finding**: the scalar reductions `L1`, `entropy`, `max|H|` are
identical for PIN and NEAR-PUZZLE (30, 1.099, 10). The **sign discriminates
them**: pin is uniformly positive; near-puzzle has uniformly negative entries
on defender pairs.

**Learned PyTorch scorer**:

- Autograd flows correctly: `pair_w[0,1].grad = +1.622` for in-tactic pair,
  `pair_w[3,4].grad = 0` for non-tactic pair
- Total forward-pass count for n=5, k=4 was 30 passes — confirms cost model

## Architecture extension — i245 Pair-Resonance Hessian Network

```
Input: simple_18 board planes
   |
   v
shared encoder phi_theta : board -> R^d  (e.g. i193's exchange/king dual-stream)
   |
   v
piece-existence saliency  s_i = |phi(P) - phi(P\i)|
   |
   v  pick top-K critical pieces (typically K=4)
   |
   v
DHPE primitive
   - 4 forward passes per (i, j) pair on top-K
   - output: K x K signed Hessian H in R^{K x K}
   |
   v
Sign-aware aggregation:
   z_pos = sum ReLU(+H_ij)  on top-pairs       (constructive interaction mass)
   z_neg = sum ReLU(-H_ij)  on top-pairs       (substitutive interaction mass)
   z_top = the K leading signed entries of H sorted by |.|
   z_base = phi(P)
   |
   v
small MLP head -> puzzle logit
```

The central claim: a position's puzzle-class is encoded in the ratio
`z_pos / (z_pos + z_neg)` plus the sparsity of `H`:

- True puzzle: `z_pos` high, `z_neg` low -> ratio -> 1
- Near-puzzle: `z_pos` and `z_neg` both significant -> ratio -> 0.5
- Neutral: both small

## Cost

For `n` pieces, top-`k` saliency selection:

- Saliency stage: `n + 1` forward passes through `phi_theta`
- Pair Hessian stage: `4 * k * (k - 1) / 2 = 2k(k-1)` passes
- Total: `n + 1 + 2k(k-1)`

For chess `n <= 32` and `k = 4`: **57 forward passes per position**.
~10x the i193 base wall-clock at scout scale. Within a 24-hour single-3070
envelope.

Compression options if budget is tight:

- Low-rank H: project pieces into `r = 8` dim and compute H on projection
- Block-diagonal H: only compute pair Hessians within piece-type buckets

## Falsifier

Run scout-scale test of i245 vs i193 baseline on canonical `puzzle_binary`
split at 173k x 12 epochs, single seed, single 3070.

**Primitive-level pass criterion**:

- Ablation A1: replace `H` with `|H|` (drop the sign). The architecture must
  lose >= 50% of any equal-bucket lift over i193 under this ablation.

**Architecture-level pass criterion**:

- `crtk_eval_bucket = equal` PR AUC: i245 >= **0.835** (i193 0.817 + 0.018)
- Aggregate test PR AUC: i245 >= 0.871 (i193 - 0.005)
- No slice regresses > 0.01 vs i193

**Fail criteria**:

- A1 ablation matches the full architecture -> drop (sign isn't load-bearing)
- Equal-bucket lift < 0.005 -> drop
- Aggregate regresses > 0.01 -> drop

## Risks

1. **Saliency-induced bias** in top-K selection. Mitigate with entropy
   regulariser on saliency; A2 ablation (learned vs deterministic geometric
   saliency from i189).
2. **Sign-collapse** under aggressive learning rate. Mitigate with auxiliary
   loss encouraging sign diversity.
3. **Hidden rebrand of Shapley interactions**. Defence: Shapley averages over
   ALL subset orderings (exponential cost). DHPE evaluates a fixed 4-vertex
   hypercube. Worth A3 ablation against 2nd-order Shapley sampler with
   matched FLOPs.
4. **Pair-saliency vs unary-saliency disagreement**. Mitigate with A4
   ablation re-selecting top-K based on row-sums of the unsigned Hessian.

## Generalisation beyond chess

Anywhere with sparse subset structure where pair-wise sub-additivity
(substitution) is meaningful: recommender systems (substitute vs complement
items), molecular interactions (cooperative vs antagonistic ligand pairs),
dynamic graphs (substitute vs reinforcing edges), redundancy detection in
sensor networks.

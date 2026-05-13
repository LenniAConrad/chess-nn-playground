# Math Thesis

Source: `ideas/research/primitives/external_37_truncated_multiset_polynomial_rook_matching_primitives.md`
(Section 1 `primitive_truncated_multiset_polynomial_pool`; first-ranked
proposal).

## Working thesis

For a position with simple_18 board tensor ``x in {0, 1}^{B x 18 x 8 x 8}``:

1. Project the per-square (piece-plane + side-to-move) descriptor to a
   tanh-bounded latent ``u_i in R^C``:

       u_i = tanh(LayerNorm(W [piece_i; stm]))

   Active token indicator: ``m_i = 1[occupancy_i > 0]``.
2. Truncate the generating polynomial

       P_b(z) = prod_{i=1}^{64} (1 + m_i u_i z)

   to degree ``K``. Initialise ``e_0 = 1``, ``e_1 = ... = e_K = 0``;
   for each token ``i``, update descending in k:

       e_k <- e_k + (m_i u_i) ⊙ e_{k-1},   k = K, K-1, ..., 1.

   Output ``E in R^{B x K x C}`` with ``E_k = e_k``.
3. Optionally normalise ``e_k`` by an estimate of ``binom(|active|, k)``
   to bound coefficient magnitudes across positions of different piece
   counts.
4. Concatenate ``E`` (flattened) with the i193 trunk joint pool feature
   and project to a scalar logit delta. Gate by trunk diagnostics +
   ``tmpp_active_mean`` + ``tmpp_coeff_norm``.

## Why this matters

Sparse-piece set operators based only on sums (DeepSets, mean / max
pooling) collapse multi-piece coalitions into the first moment. Two
defenders covering a single square contribute the same to a sum as one
defender contributing twice. Elementary-symmetric coefficients preserve
those interaction structures: ``e_2`` is large only when two distinct
active tokens both have aligned latent channels; ``e_3`` records the
three-way coalition. This is exactly the bias chess hard-negatives
exploit (one defender supports two threats, but only one can be
realised) without enumerating ``binom(n, k)`` tuples.

## What is actually proven

- The scan is exact: ``E_k = [z^k] prod_i (1 + m_i u_i z)`` componentwise.
- The K=1 ablation strictly collapses to a sum pool (with weights
  multiplied by m_i and tanh-bounded), making the ``first_order_only``
  control a clean falsifier for "is K>=2 load-bearing?".
- The gradient ``dE_k/du_i = m_i * e_{k-1}^{(-i)}`` is well-defined and
  PyTorch's autograd handles it through the recurrence.

## What is only hypothesized

That ``e_2`` / ``e_3`` carry chess-specific information not already
encoded by the i193 trunk's spatial features. The four falsifiers below
test this.

## Failure cases

1. *Hidden rebrand of DeepSets*: tested by `first_order_only`.
2. *Mask irrelevant*: tested by `uniform_mask` (and `shuffle_mask`).
3. *Coefficient explosion*: tanh-bounded latents + LayerNorm before the
   delta MLP + (optional) binomial-style normalisation in `coeff_norm`.
4. *Permutation-equivariance broken*: tested by `shuffle_tokens` —
   the scan is order-invariant for the underlying coefficient values
   (modulo per-channel LayerNorm reductions, which we keep symmetric).

## Falsifier

- `first_order_only` — primary. Collapses K from 3 to 1; the operator
  reduces to a weighted DeepSets pool. If the unablated run matches
  this control, the higher-order coalition signal is not load-bearing.
- `uniform_mask` — set ``m_i = 1`` everywhere. Tests whether the
  piece-occupancy gating matters.
- `shuffle_mask` — in-batch permutation of the occupancy mask.
  Decouples the mask from the position.
- `shuffle_tokens` — permute token order. Coefficient values are
  invariant to this; LayerNorm reductions are too. A nonzero
  difference here would expose a bug.

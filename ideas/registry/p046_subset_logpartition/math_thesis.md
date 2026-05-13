# Math Thesis

Source: `ideas/research/primitives/external_41_orbit_stabilizer_subset_logpartition_primitives.md`
(Section 2 `primitive_subset_logpartition`; the file's #2 proposal,
promoted over #1 because #1 is in the orbit/irrep family deferred to
a future symmetry batch).

## Working thesis

For a position with simple_18 board tensor:

1. Project the per-square (piece-plane + side-to-move) descriptor to a
   real-valued log-weight tensor:

       A[b, i, c] = LayerNorm(Linear([piece_i; stm]))

   Active mask: ``m[b, i] = 1[occupancy_i > 0]``. Inactive tokens are
   forced to ``A = -inf`` so they cannot contribute to any subset.
2. Run the log-semiring recurrence over the 64 squares. Boundary:

       C^{(0)}[0, c] = 0,    C^{(0)}[k > 0, c] = -inf.

   Update descending in k from K to 1:

       C^{(i)}[k, c] = logaddexp(C^{(i-1)}[k, c],
                                  C^{(i-1)}[k - 1, c] + A[i, c]).

   Output:

       Y[k, c] = C^{(N)}[k, c] = log sum_{|S|=k} exp sum_{i in S} A[i, c].
3. ``Y in R^{B x K x C}`` is clipped to [-30, 30] for head input
   bounding, LayerNormed, and concatenated with the i193 trunk joint
   pool. A delta MLP produces `primitive_delta_raw`; a gate MLP
   over (joint, active_mean, logpartition_norm) gives the sigmoid
   gate. Output:

       final_logit = base_logit + primitive_gate * primitive_delta_raw.

## Why this matters

Many tactical hard negatives differ by *how many* independent
resources exist rather than by the maximum single resource. The
log-partition records the exact log-probability mass of size-k
subsets per channel; gradients with respect to log-weights are the
marginal probability that token ``i`` belongs to such a subset.
This is the right encoding for "there are at least 2 defenders" type
features without enumerating ``binom(n, k)`` pairs explicitly.

## What is actually proven

- The log-semiring scan computes exactly the truncated elementary-
  symmetric polynomial in log-domain: the boundary plus the
  recurrence is the standard knapsack DP over (subset size, log
  weight) where the "+" operator in the semiring is replaced by
  ``logaddexp``.
- Backward through ``logaddexp`` yields the subset marginal
  ``dY[k]/dA[i] = Pr(i in S | |S|=k)`` automatically; no custom
  backward is needed.
- ``k1_only`` collapses to a simple logsumexp pool (over active
  tokens), giving a clean falsifier for "is K>=2 load-bearing?".

## What is only hypothesized

That subset-marginal information at K=2, 3 contains chess-specific
discriminative signal not already encoded by the i193 trunk and its
conv layers.

## Failure cases

1. *Hidden rebrand of logsumexp*: tested by `k1_only`.
2. *Mask irrelevant*: tested by `uniform_mask` and `shuffle_mask`.
3. *Underflow*: when log-weights have very negative channels the
   recurrence can saturate at ``-inf``. We mitigate this by LayerNorm
   on the log-weight projection and a final ``clamp(-30, 30)`` before
   the head input.
4. *Permutation-invariance broken*: tested by `shuffle_tokens` (the
   log-DP is exactly invariant to token order; LayerNorm reductions
   are symmetric).

## Falsifier

- `k1_only` — primary. Collapses K from 3 to 1; the operator
  reduces to a logsumexp pool over active log-weights.
- `uniform_mask` — set ``m = 1`` everywhere.
- `shuffle_mask` — in-batch permutation of the occupancy mask.
- `shuffle_tokens` — permute square order; coefficient values are
  invariant (regression-safe check).

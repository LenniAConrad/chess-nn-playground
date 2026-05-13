# Math Thesis

Source: `ideas/research/primitives/external_38_polynomial_ledger_grassmann_rook_primitives.md`
(Section `primitive_grassmann_rook_pool`; the file's #2 proposal,
promoted over #1 because #1 is implemented as p042).

## Working thesis

For a position with simple_18 board tensor ``x in {0, 1}^{B x 18 x 8 x 8}``:

1. Pool the i193 spatial feature map ``S in R^{B x 2C_trunk x 64}`` into
   two sets of learned query tokens:

       a_b in R^{R x D}    attacker tokens
       d_b in R^{C x D}    defender tokens

   via two independent ``BoardTokenAttention`` modules.
2. Per-token validity gates:

       m^a_{b,i} = sigmoid(W_a a_{b,i}),   m^d_{b,j} = sigmoid(W_d d_{b,j}).
3. Bipartite edge scorer:

       z_{b,i,j,h} = Bilinear(a_{b,i}, d_{b,j})_h,    h = 1, ..., H,
   tanh-bounded.
4. Truncated matching-polynomial coefficient:

       e_{b,k,h} = sum_{|S|=k, rows(S) disjoint, cols(S) disjoint}
                     prod_{(i,j) in S} m^a_i m^d_j z_{i,j,h}.

   In nilpotent-generator notation,
   ``P_h(t) = prod_{i,j} (1 + t z_{ijh} epsilon_i ^ eta_j)``
   with ``epsilon_i^2 = eta_j^2 = 0``, so any monomial that re-uses a
   row or column vanishes.
5. Closed form for K=1, K=2:

       e_1 = sum_{ij} z_{ij}                        (after masking)
       e_2 = 0.5 * ( e_1^2
                     - sum_{ij} z_{ij}^2
                     - sum_i (sum_j z_{ij})^2 - sum_{ij} z_{ij}^2
                     - sum_j (sum_i z_{ij})^2 - sum_{ij} z_{ij}^2 )

   (re-arranged to subtract row-double-counting, column-double-counting,
   and the trivial diagonal.) For K=3 we use an O(R*C) iteration that
   marginalises out one row+column at a time and applies the K=2 closed
   form on the remainder.
6. ``e in R^{B x K x H}`` is LayerNormed and concatenated with the i193
   trunk joint pool; an MLP projects to a scalar `primitive_delta_raw`.
   A separate gate MLP returns `primitive_gate`. Output:

       final_logit = base_logit + primitive_gate * primitive_delta_raw.

## Why this matters

Bipartite attention and Sinkhorn iteration both produce edge-marginal
matrices. Neither preserves the property "no row and no column may be
used twice in the same monomial" — that is enforced only after the
fact by row/column normalisation. The matching-polynomial coefficient
is the exact log-partition over disjoint edge sets of a given size,
without sampling or unrolled iteration. It is the right encoding for
"only one defender can cover this threat" / "only one attacker can
realise this tactic" interactions.

## What is actually proven

- K=1 closed form matches the trivial bilinear pool; K=2 closed form is
  derived from the algebraic identity
  ``2 * e_2 = e_1^2 - sum z^2 - row-coll - col-coll``.
- The K=3 iteration is correct because for each anchor edge ``(i, j)``,
  ``e_3 = (1/3) * sum_{(i,j)} z_{ij} * e_2^{-i,-j}`` where
  ``e_2^{-i,-j}`` is the matching coefficient on the score matrix with
  row ``i`` and column ``j`` deleted (factor 1/3 because each ordered
  triple is counted three times — once per "anchor" edge).
- `drop_exclusion` collapses to a flat elementary-symmetric pool that
  *does* double-count rows/columns, so the falsifier is meaningful.

## What is only hypothesized

That the matching-coefficient encoding outperforms the same primitive
with row/column exclusion disabled.

## Failure cases

1. *Hidden rebrand of Sinkhorn or bilinear pool*: tested by
   `drop_exclusion`.
2. *Score channels under-used*: tested by `scalar_score` (collapse to a
   single edge score).
3. *Tokens irrelevant*: tested by `shuffle_attackers` and
   `shuffle_defenders` (batch-permuted tokens decoupled from boards).
4. *Coefficient explosion*: ``score = tanh(bilinear(...))`` and a
   coefficient LayerNorm.

## Falsifier

- `drop_exclusion` — primary. Disables row/column exclusion; the scan
  collapses to an elementary-symmetric pool over flat edges (still
  truncated, but without the structural constraint). If the unablated
  run matches, the exclusion is not load-bearing.
- `scalar_score` — collapse score channels to one (mean broadcast).
  Tests whether the multi-channel edge representation is load-bearing.
- `shuffle_attackers` — in-batch permutation of attacker tokens.
- `shuffle_defenders` — in-batch permutation of defender tokens.

# Math Thesis

Source: `ideas/research/primitives/external_36_exterior_product_rank1_resolvent_primitives.md`,
rank-1 proposal `primitive_exterior_product_pool`. The rank-2 proposal
(`primitive_rank1_resolvent_pool`) is the same operator as p038
(Woodbury Set Resolver), so the same registry entry covers both.

## Working thesis

For active piece tokens `x_i` with projection `z_i = tanh(W phi(x_i))`
in `R^r`, occupancy mask `a_i in {0, 1}`, and maximum grade `R`:

    M = prod_{i in active}^{(wedge, <=R)} (1 + z_i),
    M^{(k)} = sum_{|I|=k}  bigwedge_{i in I}  z_i   in Lambda^k(R^r).

The wedge product is antisymmetric: `z_i ^ z_j = - z_j ^ z_i`, and
`z_i ^ z_i = 0`. Consequently, two collinear tokens contribute zero to
`M^{(2)}` and above. This is the key chess-relevance property: two
attackers on the same latent line are linearly dependent, and the
wedge representation cancels them automatically.

Grade-`k` component lives in `Lambda^k(R^r)` with dimension
`C(r, k)`. The output is the per-grade vectorisation
`Y = concat_{k=0..R}(vec(M^{(k)}))` of total dimension
`D_R = sum_{k=0..R} C(r, k)`. For `r = 4`, `R = 3`:
`1 + 4 + 6 + 4 = 15`.

## Incremental update semantics

The packet's defining property is that the multiset exterior-product
pool supports bounded-change deletion: in the nilpotent ring
`R[z_i] / z_i^2`, we have `(1 + z_i)^{-1} = 1 - z_i`, so removing an
event is exactly inverting its multiplication. We do not exercise the
deletion API at training time (one position per sample), but the
mathematical contract is preserved -- the forward result equals the
multivector produced by the sequence of `add(z_i)` events from an
empty multivector.

## Wedge update table

To multiply the current multivector by `(1 + z_i)` in the truncated
algebra we need: for each grade `k = R..1`, each basis multi-index
`alpha = (a_1 < ... < a_{k-1})`, and each scalar coordinate `j in 0..r-1`,

- if `j in alpha`: the wedge `e_alpha ^ e_j = 0`;
- else: it lands at the canonical grade-`k` basis index
  `beta = sorted(alpha + (j,))` with sign `(-1)^{count(a > j)}`,
  where the count is the number of elements of `alpha` strictly larger
  than `j` (the number of swaps to bring `j` into sorted order).

We precompute `target_basis_{k}` and `target_sign_{k}` as static
buffers at construction time. The forward pass is then a
`scatter_add_` per grade, per active token.

## Architecture-level claim

    final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * delta(concat(M^{(0)}, ..., M^{(R)}))

with the gate initialised closed (`gate_init = -2.0`).

## Falsifiers

- Primitive-level: `shuffle_grades_high` (in-batch permutation of
  `M^{(k)}` for `k >= 2`) must lose the slice lift.
- `first_order_only` (zero `M^{(>=2)}`) must lose the higher-order
  cancellation component -- if the unablated run does not beat A2,
  the wedge structure adds nothing over a sum pool.
- Architecture-level: p041 must beat i193 on its declared slice
  (grade-2 magnitude above median, i.e. positions where the wedge
  cancellation is informative) without regressing aggregate PR AUC.

## Why this is not p024 (elementary-symmetric polynomial accumulator)

p024 uses Hadamard products (commutative): `(z_i, z_j) -> z_i (.) z_j`
is symmetric, so collinear tokens *add* their interaction strength.
p041 uses wedge products (antisymmetric): `(z_i, z_j) -> z_i ^ z_j`
is alternating, so collinear tokens *cancel*. Same coefficient-style
recurrence, fundamentally different algebra.

## Why this is not p038 (Woodbury Set Resolver)

p038 downweights collinear tokens through the *inverse precision*
`(lambda I + sum_i z_i z_i^T)^{-1}` -- a continuous (smoothed)
cancellation that depends on the regulariser `lambda`. p041 cancels
collinear tokens *exactly* through the alternating wedge. The two
primitives are complementary -- p038 is cheaper for large `r` (cubic in
`r`), p041 is cheaper for small `r` (factorial in `r`).

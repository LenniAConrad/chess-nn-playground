# Math Thesis

Source: `ideas/research/primitives/external_13_reversible_delta_kernel_occlusion_transport.md`,
rank-1 proposal `primitive_01_reversible_delta_kernel_memory`.

## Working thesis

A dynamic set state `(M, z)` is maintained over the active piece set on
the board:

```
M = sum_i phi(u_i) nu(u_i)^T,    z = sum_i phi(u_i)
```

with `u_i in R^d` the token of the i-th active piece (piece type + side
+ square), `phi(.) = elu(.) + 1` a positive feature map (the standard
linear-attention nonlinearity), and `nu(.)` a learnable value
projection. Each query `q` reads the memory through the
linear-attention-style normalisation

```
Y_q = (phi(q)^T M) / (phi(q)^T z + epsilon).
```

The primitive's defining property is its update API: a signed event
`(s, u_e)` with `s in {-1, +1}` updates `M` and `z` exactly:

```
M' = M + s * phi(u_e) nu(u_e)^T
z' = z + s * phi(u_e).
```

`remove(u_e)` is exactly the inverse of `add(u_e)`, so an engine make/
unmake loop can keep the memory in sync with the board in O(events)
time rather than O(|active|).

At training time we observe one static board per sample, so the forward
pass builds `(M, z)` once over the active piece set. The output equals
the result of running the corresponding sequence of insert events from
an empty memory.

## Architecture-level claim

```
final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * delta(rdkm_readout(x))
```

where `joint` is the i193 pooled joint feature (stop-gradient is *not*
applied to the gate input, but it could be; see `architecture.md`).
The gate is initialised near zero so the head starts as a no-op and
must learn to fire on positions that need second-order kernel-memory
signal.

## Falsifier

- Primitive-level: `shuffle_tokens` (in-batch permutation of the
  per-square token tensor) must lose the slice lift versus the
  unablated run. If it matches, the kernel memory carries no signal
  beyond what the trunk already encodes.
- `zero_memory` (drop ``M, z`` to zero) plus `uniform_query` (replace
  the trunk-derived queries with a constant) further isolate which
  pieces of the operator are load-bearing.
- Architecture-level: p019 must beat i193 on its declared slice
  (king-piece distance / pinned-piece patterns) without regressing
  aggregate PR AUC. The shuffle ablation must lose >=70% of that lift.

## Why this is not "just linear attention"

Linear attention is a *sequence* operator with causal mask and ordered
recurrence. The kernel memory here is an *unordered set* operator: the
inputs are the active pieces of the board, no order, and the API has
explicit signed deletion. Linear attention's appending semantics do
not directly support exact removal of a previously written key/value
pair without recomputing the full prefix.

The packet's audit notes Gated DeltaNet and Kimi Delta Attention as
related families; the implementation here is honest about that overlap
and only claims the unordered signed-edit kernel-memory operator, not
"first kernel memory ever".

## Why this is not NNUE

Stockfish NNUE maintains a *first-order* accumulator (column sums of
the input layer). p019 stores a *second-order* outer product memory
`phi nu^T`. The two are complementary: NNUE captures sparse first-order
piece-square statistics; p019 captures pairwise piece-piece interactions
through the kernel-attention query mechanism.

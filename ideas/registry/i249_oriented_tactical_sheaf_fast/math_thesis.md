# Math Thesis

Oriented Tactical Sheaf Laplacian (Fast) -- i249.

i249 is a pure execution rewrite of i018. It does not propose a new chess
signal, loss, input, or head. Its claim is that i018's sheaf Laplacian update can
be evaluated without materializing every edge residual.

## Inherited Object

The cell complex, relation masks, stalks, restriction maps, signs, gates, heat
step, pooling, and classifier are inherited from i018:

- 64 square 0-cells;
- 12 typed tactical relations;
- stalk dimension `s` (default 8);
- learned `rho_src[r]`, `rho_dst[r]`;
- fixed signs `sigma_r`;
- bounded gates `g_r` and heat-step scale `eta`;
- the same triad-defect and readout diagnostics.

## Algebraic Rewrite

For one relation, i018 computes:

```text
src_i = z_i rho_src
dst_j = z_j rho_dst
residual_ij = dst_j - sigma src_i
weighted_ij = g W_ij residual_ij
```

The node update is:

```text
g * sigma * sum_j weighted_ij rho_src.T
-g *        sum_i weighted_ij rho_dst.T
```

Expanding before materialization gives:

```text
source_pre_i = sigma * (W @ dst)_i - out_degree_i * src_i
target_pre_j = sigma * (W.T @ src)_j - in_degree_j * dst_j
```

and therefore:

```text
update = g * source_pre rho_src.T + g * target_pre rho_dst.T
```

This is the same sheaf heat update, just evaluated with batched matrix products
and endpoint degrees instead of a `(B, 64, 64, s)` residual tensor.

The energy diagnostic is also expanded:

```text
sum_ij W_ij ||dst_j - sigma src_i||^2
= sum_i out_i ||src_i||^2
  + sum_j in_j ||dst_j||^2
  - 2 sigma sum_i src_i dot (W @ dst)_i
```

The gate and denominator are applied exactly as in i018.

## Equivalence Claim

For shared weights and fixed precision mode:

```text
forward_i249(x; theta) ~= forward_i018(x; theta)
grad_i249(loss; theta) ~= grad_i018(loss; theta)
```

The equality is up to floating-point reduction order. Current checks found
logits within about `1e-7` and gradients within about `4.5e-8`.

## Speed Claim

The original i018 bottleneck was memory traffic from residual materialization
and reductions inside a 12-relation loop. i249 replaces that with:

- two relation-batched stalk projections;
- two relation-batched `64x64 @ 64xs` products;
- two relation-specific backsweeps;
- one optional compiled static forward.

On the local RTX 4070 Laptop GPU, this reduced base-scale eager batch-256
forward latency from about `71.7 ms` to about `13.5 ms`; compiled with TF32/high
precision it reached about `5.9 ms`. The optional eval-only FP16 autocast path
reached about `4.23 ms` at batch 256, with random-batch logit drift around
`2e-4` versus the FP32/TF32 i249 path.

## Falsifiers

- Numerical: shared-weight logits diverge by more than `1e-5`, or checked
  gradients diverge by more than `1e-7`.
- Accuracy: matched paper-grade runs fall outside i018's normal seed noise.
- Speed: the same host GPU does not show a clear throughput gain over i018.

If any falsifier trips, keep i018 as the canonical architecture and treat i249
as a failed execution rewrite rather than a new model.

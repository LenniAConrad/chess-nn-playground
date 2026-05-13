# Math Thesis

Source: `ideas/research/primitives/external_20_event_symmetric_sparse_scatter_ray_scan.md`,
rank-1 proposal `primitive_event_symmetric_accumulator`.

## Working thesis

For active piece tokens `u_i in R^d` (`i in S`) and order `R >= 1`,
the elementary symmetric polynomial states under Hadamard product
are:

```
E^{(0)} = 1   (in R^d, the multiplicative identity for Hadamard)
E^{(r)} = sum_{i_1 < ... < i_r in S} u_{i_1} (.) ... (.) u_{i_r}
```

The streaming recurrence (Newton-style) maintains these states with
exact insert and delete events:

```
add(u):
  for r = R, R-1, ..., 1:
    E^{(r)} <- E^{(r)} + u (.) E^{(r-1)}

remove(u):
  tilde E^{(0)} = 1
  for r = 1, 2, ..., R:
    tilde E^{(r)} = E^{(r)} - u (.) tilde E^{(r-1)}
```

`remove(u)` is exactly the inverse of `add(u)`, so the engine make/
unmake loop can keep the state in sync in `O(R d)` per event. At
training time we run the static `add` series over the full active
set; the static and incremental computations agree by construction.

## Architecture-level claim

```
final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * MLP[E^{(1)}; ...; E^{(R)}]
```

The gate is initialised near zero so the head starts as a no-op. The
default order is `R = 2`; the implementation also supports `R = 3`
(third-order interactions).

## Falsifier

- Primitive-level: `first_order_only` keeps only `E^{(1)}` (the
  EmbeddingBag-equivalent sum); `second_order_only` keeps only
  `E^{(2)}`. If `first_order_only` matches `none`, the higher-order
  states are not load-bearing.
- `shuffle_higher_orders` (in-batch permutation of `E^{(>=2)}`)
  decouples higher orders from positions.
- Architecture-level: p024 must beat i193 on the declared third-order
  slice (knight fork, double-attack, discovered-attack triple)
  without regressing aggregate PR AUC.

## Why this is not just `EmbeddingBag` plus polynomial pooling

`EmbeddingBag` is exactly `E^{(1)}` for binary feature presence.
Polynomial pooling is a stateless static operator. The defensible
*primitive* claim is the reversible event API: every `add` has an
exact inverse `remove` with the same cost, and the static recompute
agrees with any sequence of `add` events from an empty state by
construction. We name `EmbeddingBag` and polynomial pooling
explicitly in the source packet's self-audit.

For `R = 2` the static form has a closed-form factorisation:

```
E^{(2)} = (1/2) * ((sum_i u_i) (.) (sum_i u_i) - sum_i u_i (.) u_i)
```

which is the same FM-style identity p022 uses with `U_i = V_i = u_i`.
For `R >= 3` the streaming recurrence is the most efficient way to
compute the elementary symmetrics without enumerating triples or
applying Newton's identities to power-sums (which need divisions and
are less numerically stable).

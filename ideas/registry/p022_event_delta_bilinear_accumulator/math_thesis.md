# Math Thesis

Source: `ideas/research/primitives/external_18_delta_bilinear_ray_blocked_segment_attention.md`,
rank-1 proposal `primitive_delta_bilinear_accumulator` (Event-Delta
Bilinear Accumulator).

## Working thesis

For active piece tokens `u_i` (indexed by occupied squares) with two
learned projections `U_i = W_U u_i in R^d`, `V_i = W_V u_i in R^d`:

```
A = sum_i U_i in R^d
B = sum_i V_i in R^d
Q = sum_{i<j} (U_i (.) V_j + U_j (.) V_i) in R^d
```

The pair term `Q` has a closed-form factorisation (the standard
factorisation-machine identity, generalised to vector-valued tokens):

```
Q = (sum_i U_i) (.) (sum_j V_j) - sum_i U_i (.) V_i
  = A (.) B - sum_i (U_i (.) V_i)
```

so the static forward at one position costs `O(|S| d)` time instead
of the naive `O(|S|^2 d)`. The output is

```
Y = MLP[A; B; Q] in R
```

## Architecture-level claim

```
final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * MLP[A; B; Q]
```

The gate is initialised near zero so the head starts as a no-op.

## Event-update API (deferred)

The stateful primitive maintains `(A, B, sum_i U_i (.) V_i)` and
updates them in `O(d)` per insert/delete event:

```
add(u): A += U(u); B += V(u); P += U(u) (.) V(u)
remove(u): A -= U(u); B -= V(u); P -= U(u) (.) V(u)
```

`remove` is the exact inverse of `add`. `Q = A (.) B - P` is read out
on demand. Per-event cost is `O(d)`, comparable to NNUE's
HalfKA-style accumulator update. This API is not exercised at
training time because the trainer feeds one static board per sample;
the engine inference path would use it.

## Falsifier

- Primitive-level: `first_order_only` (drop the pair term `Q`) must
  hurt the declared slice more than i193 alone. If the pair term is
  not load-bearing, this collapses to NNUE-style accumulation.
- `shuffle_pair_term` (in-batch permutation of `Q`) decouples the
  pair signal from positions.
- Architecture-level: p022 must beat i193 on slices that depend on
  second-order interactions without regressing aggregate PR AUC.

## Why this is not Factorisation Machines (verbatim)

FMs (Rendle 2010) factorise pair interactions of binary features. The
static math here is the FM identity. The defensible *primitive* claim
is the stateful event-update API (`add` / `remove` with `O(d)` cost
per event and a persistent accumulator state), not the FM algebra
itself. We name FMs explicitly in the source packet's self-audit and
in the implementation notes; the implementation is honest that the
static recompute is the FM identity.

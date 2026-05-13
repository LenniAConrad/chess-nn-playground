# Math Thesis

Source: `ideas/research/primitives/external_16_ray_blocked_delta_pair_legal_edge_reduce.md`,
rank-1 proposal `primitive_ray_blocked_scan` (Occlusion Semiring Ray
Scan).

## Working thesis

For each square `s`, direction `r`, ordered ray cells
`c_{s,r,1..L}`, and per-square occupancy `O`:

```
T_{b, s, r, l} = prod_{q < l} (1 - O_{b, c_{s,r,q}})
y_{b, s} = sum_r sum_{l=1..L} T_{b, s, r, l} * A_r * x_{b, c_{s,r,l}}
```

`T` is the **exclusive prefix transmittance** along the ray: cell `l`
is reachable from `s` only if every previous cell on that ray is
unoccupied. `A_r` is a per-direction projection (8 distinct linear
maps, one per queen direction).

We compute `T` in log-domain via `cumsum(log(1 - O))` shifted by one
position to make the prefix exclusive. A small `log_eps = 1e-4` lower
bound on `(1 - O)` prevents `log(0)` when a square is occupied (which
is the expected case for a blocker -- the resulting `T = 0` zeroes
the cell as required).

## Architecture-level claim

`y_{b, s}` is mean-pooled across squares to a `(B, 8 * hidden_dim)`
vector, projected through a small MLP, and added to the i193 base
logit via a sigmoid gate:

```
final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * delta(osrs_readout(x))
```

The gate is initialised near zero so the head starts as a no-op.

## Falsifier

- Primitive-level: `zero_occupancy` (treat the board as empty, so
  `T = 1` everywhere) and `uniform_occupancy` (treat every square as
  occupied, so only step 1 has `T > 0`) -- if neither hurts the
  declared slice, the transmittance is not load-bearing.
- `isotropic_A` (share the projection across directions) tests whether
  per-direction parameters carry signal.
- Architecture-level: p021 must beat i193 on x-ray / pin / skewer
  slices without regressing aggregate PR AUC.

## Why this is not Conv2d / masked attention / Mamba / p020 / p023

- Conv2d has a fixed local kernel and no per-cell visibility weight.
- Masked attention takes an external mask; the prefix-product
  transmittance here is generated *inside* the operator.
- Mamba's selective SSM is a sequence recurrence; here there are 8
  rays per source square and the transmittance is a non-recurrent
  prefix product, not an input-conditioned state transition.
- p020 uses a *recurrence with hard reset* over hidden states; the
  exclusive prefix product is conceptually different (it does not
  carry hidden state between ray cells -- it weighs each cell's own
  projection directly).
- p023 uses a *backward semiring recurrence* (gate factor
  `(1 - O_{t+1})` on `h_{t+1}`); the forward exclusive prefix product
  here aggregates outward from the source with a different gradient
  topology.

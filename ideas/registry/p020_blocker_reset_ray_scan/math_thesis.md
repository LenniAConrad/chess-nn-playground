# Math Thesis

Source: `ideas/research/primitives/external_15_blocker_reset_edit_delta_fastweight.md`,
rank-1 proposal `primitive_blocker_reset_scan` (Blocker-Reset Ray
Scan).

## Working thesis

For each board square `s in 0..63`, direction `d in 0..7`, ordered
ray cells `s_t = ray(s, d, t)` for `t = 0..L_max`, and per-direction
decay parameter `lambda_d in (0, 1)^h`:

```
h_{s, d, 0} = U * x_s                                    # source step
h_{s, d, t} = U * x_{s_t} + (1 - O_{s_t}) (.) lambda_d (.) h_{s, d, t-1}
y_{s, d}    = V * (1/T) * sum_{t=0..T} h_{s, d, t}
```

where `O` is the per-square occupancy from the simple_18 piece planes
(stop-gradient as a binary), `U` is a token-to-hidden projection,
`V` is a hidden-to-output projection, and `T` is the number of valid
ray steps from `s` along `d`. Off-board steps are masked.

The defining property is the **hard reset gate** `(1 - O_{s_t})`: a
blocker at step `t` zeroes the contribution of all previous steps to
`h_{s, d, t+1}` and beyond. This matches the chess sliding-piece
invariant exactly -- a rook ray, bishop ray, or queen ray stops at the
first blocker, and pin / x-ray geometry is the difference between the
ray that *would* be visible if a blocker were removed and the one that
*is* visible.

## Architecture-level claim

The per-direction ray output `y_{s, d}` is mean-pooled across squares
to a `(B, 8 * token_dim)` vector and projected to a scalar logit
delta. The final logit is

```
final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * delta(brrs_readout(x))
```

The gate is initialised near zero so the head starts as a no-op.

## Falsifier

- Primitive-level: `zero_blocker` (ignore occupancy gate, run the full
  scan) must beat the unablated p020 by less than the operator's
  declared lift. If it matches, the blocker reset is not load-bearing.
- `uniform_blocker` (treat every square as occupied) collapses the
  scan to just the source-step contribution -- a check that the
  recurrence depth carries signal at all.
- Architecture-level: p020 must beat i193 on slices that depend on
  sliding-piece vision without regressing aggregate PR AUC.

## Why this is not Conv2d / masked attention / Mamba

- Conv2d has a fixed local kernel; the segment depth here is content-
  dependent (the blocker can be anywhere on the ray).
- Masked attention takes an external mask; the blocker mask here is
  generated *inside* the operator from occupancy.
- Mamba's selective SSM is a one-dimensional sequence recurrence; here
  there are 8 ray sequences per source square, all coupled through
  the shared source-step token.

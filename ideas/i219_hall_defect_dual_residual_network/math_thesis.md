# Math Thesis

## Working thesis

A near-puzzle has defender obligations that an exact deficiency count calls
feasible, but whose Lagrangian dual trajectory is fragile. A true puzzle
creates a hard infeasibility whose dual trajectory blows up under a few
projected ascent steps.

## Setup

Let `A in [0, 1]^{m x n}` be a learned soft incidence matrix between `m`
defender obligations and `n` candidate defender resources, and let
`demand in R^m_+`, `cost in R^n_+` be learned strictly positive vectors.
The relaxed defender-covering LP is

```text
minimize    cost^T z
subject to  A z >= demand,
            0 <= z <= 1.
```

We unroll `T = 5` projected dual-ascent updates

```text
z       <- clip(z - eta (cost - A^T lambda), 0, 1)
lambda  <- max(0, lambda + eta (demand - A z))
```

and pool a fixed-size summary of the per-step primal violation, current
objective, complementarity, and final primal/dual norms as the discriminative
trajectory feature.

## Claim

The trajectory features of the unrolled dual ascent contain Lagrangian-profile
information that final-only Hall deficiency counts discard. Trajectory
residuals should add to a Hall-style baseline at puzzle-vs-near-puzzle
separation.

## Falsifiers

- `final_only`: classify only from the final `z`, `lambda`, and primal
  violation.
- `random_dual_steps`: randomize the dual update direction.
- `degree_matched_rewire`: rewire the incidence matrix preserving row and
  column sums.

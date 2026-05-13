# Ablations

p021 supports five ablation modes via `model.ablation`. The primary
falsifier is `zero_occupancy` -- it removes the operator's defining
visibility weight (the prefix transmittance).

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `zero_occupancy` | Treat the board as empty (`T = 1` everywhere). **The primary falsifier.** If A1 matches `none` on the declared slice, the transmittance is not load-bearing. |
| A2 | `uniform_occupancy` | Treat every square as occupied (`T > 0` only at step 1). Tests that ray depth carries signal. |
| A3 | `isotropic_A` | Share the projection across all 8 directions. Tests whether direction-specific parameters help. |
| A4 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A5 | `trunk_only` | Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p021 >= i193 - 0.005, AND
- declared sliding-piece slice PR AUC of unablated p021 >= i193 + 0.04, AND
- A1 (`zero_occupancy`) loses >= 70% of the slice lift, AND
- training throughput drop versus i193 < 25%.

Drop if any condition fails. Drop especially if A1 matches `none`.

## Out-of-scope ablations (future)

- Use `cumprod` instead of `log + cumsum`.
- Add a learnable per-direction scale on top of `A_r`.
- Use soft occupancy from a small CNN instead of the rule-derived
  binary occupancy.

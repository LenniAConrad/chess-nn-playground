# Ablations

p020 supports four ablation modes via `model.ablation`. The primary
falsifier is `zero_blocker` -- it disables the operator's defining
property (the hard reset).

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `zero_blocker` | Replace `(1 - O)` with `1`. The scan runs the full ray length regardless of blockers. **The primary falsifier.** If A1 matches `none` on the declared slice, the blocker reset is not load-bearing. |
| A2 | `uniform_blocker` | Replace `(1 - O)` with `0`. The scan only sees the source token; the recurrence depth contributes nothing. Sanity check that depth matters. |
| A3 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. Sanity check that wrapping the trunk did not regress the baseline. |
| A4 | `trunk_only` | Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p020 >= i193 - 0.005, AND
- declared sliding-piece slice PR AUC of unablated p020 >= i193 + 0.04, AND
- A1 (`zero_blocker`) loses >= 70% of the slice lift, AND
- training throughput drop versus i193 < 25%.

Drop if any condition fails. Drop especially if A1 matches `none` --
that means the head learned a generic ray feature that does not
require the blocker reset, which is exactly what the primitive claims
to be load-bearing.

## Out-of-scope ablations (future)

- Per-direction decay sharing (tie `lambda` across directions).
- Replace the mean-pool ray summary with a last-step or attention-pool
  summary.
- Vary the maximum ray length.

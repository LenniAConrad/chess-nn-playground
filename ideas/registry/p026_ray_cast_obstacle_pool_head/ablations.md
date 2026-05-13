# Ablations

`p026` exposes the shared primitive-head ablations plus three RayPool-
specific controls. Primary falsifier is `drop_occlusion` — every
promotion run must include this matched control.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `drop_occlusion` | Ignore the blocker mask. Rays decay through occupied squares. **Primary falsifier.** If the architecture matches A1 on long-range slices, occlusion termination is not load-bearing and the primitive is dropped. |
| A2 | `shuffle_directions` | Random permutation of the 8 directions (the `gamma` vector is shuffled too). Tests whether direction-specific learning is load-bearing. |
| A3 | `zero_rays` | Replace the pooled feature vector with zeros. Tests whether the trunk diagnostics in the fusion vector are doing all the work. |
| A4 | `zero_delta` | Force `primitive_delta = 0`. Recovers i193 baseline. |
| A5 | `disable_gate` | Hold the gate at 1.0. Tests whether the learned gate is load-bearing. |
| A6 | `trunk_only` | Zero gate and delta together. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p026 >= i193 - 0.005, **and**
- the long-range tactical slice (back-rank mate, skewer, battery, pin)
  PR AUC of unablated p026 >= i193 + 0.02, **and**
- A1 (`drop_occlusion`) loses >= 70% of the long-range lift, **and**
- A2 (`shuffle_directions`) loses >= 50% of the long-range lift.

Drop if any condition fails. Drop especially if A1 matches the
unablated run — that means the head ignored occlusion at training time
even though the architecture exposes it.

## Out-of-scope ablations (future)

- Per-direction `gamma` freezing (initialise from chess-prior values).
- Replace mean-pool with attention pooling over directions.

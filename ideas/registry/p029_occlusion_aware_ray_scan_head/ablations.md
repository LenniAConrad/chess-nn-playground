# Ablations

`p029` exposes the shared primitive-head ablations plus three OARS-
specific controls. Primary falsifier is `disable_blocker_gate` —
every promotion run must include this matched control.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `disable_blocker_gate` | Force the blocker gate to 1.0. **Primary falsifier.** If the architecture matches A1 on the target slice, the selective gating is not load-bearing and the head collapses to a (variant of) plain prefix sum. |
| A2 | `shuffle_directions` | Random per-pass permutation of the 8 directions. Tests whether direction-specific learning is load-bearing. |
| A3 | `zero_oars_features` | Replace the pooled output with zeros. Tests the trunk-diagnostics contribution. |
| A4 | `zero_delta` | Force `primitive_delta = 0`. Recovers i193 baseline. |
| A5 | `disable_gate` | Hold the *output* gate at 1.0 (distinct from the blocker gate). |
| A6 | `trunk_only` | Zero gate and delta together. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p029 >= i193 - 0.005, **and**
- the long-range / x-ray / pinned-piece slice PR AUC of unablated
  p029 >= i193 + 0.02, **and**
- A1 (`disable_blocker_gate`) loses >= 70% of the target slice lift,
  **and**
- the unablated run beats `p026` RayPool on the same slice (otherwise
  prefer the cheaper RayPool head).

Drop if any condition fails. Drop especially if A1 matches the
unablated run — selective gating is the *defining* feature of OARS.

## Out-of-scope ablations (future)

- State-dependent blocker gate (currently the gate is conditioned on
  the raw per-square features; the full spec calls for conditioning
  on the running state).
- Parallel-scan implementation (currently sequential; the Mamba-style
  parallel selective scan would be a follow-on once the eager version
  is validated).

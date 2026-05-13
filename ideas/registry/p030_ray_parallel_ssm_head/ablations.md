# Ablations

`p030` exposes the shared primitive-head ablations plus four Ray-SSM
specific controls. Primary falsifiers are `disable_selective_A` and
`disable_selective_B` — every promotion run must include both matched
controls.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `disable_selective_A` | Force A to a constant 0.5. **Primary falsifier.** Tests whether the input-conditioned retention is load-bearing. |
| A2 | `disable_selective_B` | Force B to a constant 0.5. **Primary falsifier.** Tests whether the input-conditioned injection is load-bearing. |
| A3 | `no_directional_C` | Replace each per-direction `C` with the mean across directions. Tests whether direction-specific read-out is load-bearing. |
| A4 | `zero_ssm_features` | Replace the pooled SSM output with zeros. Tests the trunk-diagnostics contribution. |
| A5 | `zero_delta` | Force `primitive_delta = 0`. Recovers i193 baseline. |
| A6 | `disable_gate` | Hold the *output* gate at 1.0. |
| A7 | `trunk_only` | Zero gate and delta together. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p030 >= i193 - 0.005, **and**
- the long-range mixing slice PR AUC of unablated p030 >= i193 +
  0.02, **and**
- A1 (`disable_selective_A`) **or** A2 (`disable_selective_B`)
  loses >= 70% of the lift, **and**
- the unablated run beats both `p026` RayPool and `p029` OARS on
  the same slice (otherwise prefer the cheaper / simpler head).

Drop if any condition fails. Drop especially if both A1 and A2
match the unablated run — that means the head behaved as a constant
recurrence in practice.

## Out-of-scope ablations (future)

- Parallel selective scan (Mamba-style fused kernel).
- Per-square C (currently per-direction-only; the full spec calls
  for `C_{i, d}` indexed by both).
- Cross-channel A and B (currently diagonal).

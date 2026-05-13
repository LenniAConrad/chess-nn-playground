# Ablations

`p028` exposes the shared primitive-head ablations plus four ILA-
specific controls. Primary falsifier is `zero_king_accumulator` —
every promotion run must include this matched control.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `zero_king_accumulator` | Hold `h_king = 0`. **Primary falsifier.** If the architecture matches A1, the king-anchored accumulator is not load-bearing and p028 collapses to p025 with extra weight. |
| A2 | `zero_global_accumulator` | Hold `h_global = 0`. Tests whether the global accumulator is load-bearing. |
| A3 | `linear_only` | Skip the `phi` non-linearity. Tests whether the non-linear lift is load-bearing. If A3 matches the unablated run, p028 is *equivalent to* p025 / IDL + king-anchored embedding. |
| A4 | `shuffle_square_order` | Random column permutation of the indicator. Decouples per-square structure from real squares. |
| A5 | `zero_delta` | Force `primitive_delta = 0`. Recovers i193 baseline. |
| A6 | `disable_gate` | Hold the gate at 1.0. Tests whether the gate is load-bearing. |
| A7 | `trunk_only` | Zero gate and delta together. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p028 >= i193 - 0.005, **and**
- the king-safety / king-zone slice PR AUC of unablated p028
  >= i193 + 0.02, **and**
- A1 (`zero_king_accumulator`) loses >= 70% of the king-related lift,
  **and**
- A3 (`linear_only`) loses meaningful aggregate signal (otherwise
  the linear-only test is equivalent to p025 and we should drop p028
  in favour of the cheaper head).

Drop if any condition fails. Drop especially if A1 matches the
unablated run — the king-anchored embedding is the *defining* feature
of HalfKA-style ILA.

## Out-of-scope ablations (future)

- Replace the king-anchored embedding with a piece-king relative-
  square encoding (saves memory, slightly less expressive).
- Two-king (own + enemy) anchored embedding.
- Frozen `phi` init (initialise as identity and let optimiser unfreeze).

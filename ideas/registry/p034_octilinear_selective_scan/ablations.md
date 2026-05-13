# Ablations

p034 supports seven ablation modes via `model.ablation`. The primary
falsifier is `single_direction` -- every promotion run must include
this matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `single_direction` | Run only the E direction; zero the other seven. **Primary falsifier.** If A1 matches the unablated run, the eight-direction decomposition is not load-bearing. |
| A2 | `fixed_transition` | ``A_k`` becomes a data-independent learned parameter (no data dependence on the per-step input). Tests whether Mamba-style selectivity is load-bearing. |
| A3 | `shuffle_features` | Permute seed features across the batch. Tests rule-feature load-bearing. |
| A4 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A5 | `trunk_only` | Same as A4 (semantic alias). |
| A6 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p034 >= i193 - 0.005, AND
- the target-slice (long-range ray coordination tactics: open files,
  diagonal pins, far-piece coordination) PR AUC lifts at least +0.02
  over i193, AND
- A1 (`single_direction`) loses >= 50% of that lift, AND
- A2 (`fixed_transition`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Bidirectional pair sharing*: tie E/W, N/S, NE/SW, NW/SE parameters
  to halve the head's parameter count.
- *Parallel-scan kernel*: replace the Python loop with a Triton /
  CUDA parallel scan; required for wall-clock parity with Mamba.
- *Per-piece-type selectivity*: condition ``A_k`` on the piece type at
  each step (currently the gate sees only the seed feature).

Run these only after the primary falsifier (`single_direction`)
passes.

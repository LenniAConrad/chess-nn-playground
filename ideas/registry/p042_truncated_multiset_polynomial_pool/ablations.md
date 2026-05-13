# Ablations

p042 supports seven ablation modes via `model.ablation`. The primary
falsifier is `first_order_only` -- every promotion run must include
this matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `first_order_only` | Set effective K := 1. **Primary falsifier.** If A1 matches the unablated run, the higher-order coalition structure (e_2, e_3) is not load-bearing -- the operator collapses to a DeepSets-style weighted sum pool. |
| A2 | `uniform_mask` | Replace the occupancy mask with all-ones (so empty squares contribute too). Tests whether the chess-rule occupancy mask carries signal. |
| A3 | `shuffle_mask` | In-batch permutation of the occupancy mask. Decouples the mask from the position. |
| A4 | `shuffle_tokens` | Permute square (token) order before the scan. Coefficient values are *exactly* invariant; reductions in the head are symmetric. A non-zero delta diff would expose a bug. |
| A5 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A6 | `trunk_only` | Same as A5 (semantic alias). |
| A7 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p042 >= i193 - 0.005, AND
- the target-slice ("multi-piece coalition / hanging defender" puzzles
  per the source primitive) PR AUC lifts at least +0.02 over i193, AND
- A1 (`first_order_only`) loses >= 50% of that lift (i.e. K>=2 is
  load-bearing), AND
- A2 (`uniform_mask`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Per-piece-type token projection*: condition ``u_i`` on piece type.
- *Log-domain coefficient scan*: replace the multiplicative recurrence
  with a log-semiring scan for K >= 4. Numerically more stable but
  requires a different backward.
- *Delete-update path*: implement the spec's O(qKd) bounded-change
  update via polynomial division when NNUE-style bounded-edit
  inference is available.

Run these only after the primary falsifier (`first_order_only`)
passes.

# Ablations

p046 supports seven ablation modes via `model.ablation`. The primary
falsifier is `k1_only` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `k1_only` | Set effective K := 1. **Primary falsifier.** If A1 matches the unablated run, the higher-order log-partition structure (Y_2, Y_3) is not load-bearing -- the operator collapses to a logsumexp pool. |
| A2 | `uniform_mask` | Replace the occupancy mask with all-ones. Tests whether the chess-rule occupancy mask carries signal. |
| A3 | `shuffle_mask` | In-batch permutation of the occupancy mask. Decouples the mask from the position. |
| A4 | `shuffle_tokens` | Permute square (token) order before the scan. Log-partition values are *exactly* invariant; LayerNorm + delta head reductions are symmetric. A non-zero delta diff would expose a bug. |
| A5 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A6 | `trunk_only` | Same as A5 (semantic alias). |
| A7 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p046 >= i193 - 0.005, AND
- the target-slice ("k-defenders threshold / multi-attacker race"
  puzzles per the source primitive) PR AUC lifts at least +0.02 over
  i193, AND
- A1 (`k1_only`) loses >= 50% of that lift, AND
- A2 (`uniform_mask`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Log-domain edit update*: bounded-change inference via polynomial
  factor removal.
- *Mixed-degree fusion*: combine the multiplicative-domain p042
  output with the log-domain p046 output to see whether they carry
  complementary information.

Run these only after the primary falsifier (`k1_only`) passes.

# Ablations

p033 supports seven ablation modes via `model.ablation`. The primary
falsifier is `shared_kernel` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shared_kernel` | Collapse all 6 move types to a single shared projection. **Primary falsifier.** If A1 matches the unablated run, the move-type weight sharing is not load-bearing. |
| A2 | `scalar_per_type` | Replace per-type matrix with per-type scalar gain. Tests whether matrix capacity beyond a scalar is load-bearing. |
| A3 | `shuffle_features` | Permute seed features across the batch. Tests rule-feature load-bearing. |
| A4 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A5 | `trunk_only` | Same as A4 (semantic alias). |
| A6 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p033 >= i193 - 0.005, AND
- the target-slice ("long-range tactic" puzzles per the source primitive)
  PR AUC lifts at least +0.02 over i193, AND
- A1 (`shared_kernel`) loses >= 50% of that lift, AND
- A2 (`scalar_per_type`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- Per-piece-type input feature (the spec mentions ``W_t @ X_j * indicator(piece_at_j)``);
  would require splitting the seed feature into per-piece channels.
- Blocker-resolved variant (essentially p032 DAG); the contrast is
  already covered by the existing p032 / p033 dual.
- Sparse gather-scatter CUDA kernel for wall-clock wins.

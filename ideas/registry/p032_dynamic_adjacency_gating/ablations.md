# Ablations

p032 supports eight ablation modes via `model.ablation`. The primary
falsifier is `single_move_type` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `single_move_type` | Collapse all 8 move types to a single shared projection. **Primary falsifier.** If A1 matches the unablated run, the per-type weight specialisation is not load-bearing. |
| A2 | `soft_mask` | Replace binary mask with sigmoid(2 * (A - 0.5)). Tests whether the hard-mask story is load-bearing. |
| A3 | `uniform_adjacency` | Replace adjacency with all-ones (minus identity). Tests whether the rule-derived edges matter. |
| A4 | `shuffle_adjacency` | In-batch permutation of the legal-move graph. Tests rule-feature load-bearing. |
| A5 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A6 | `trunk_only` | Same as A5 (semantic alias). |
| A7 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p032 >= i193 - 0.005, AND
- the target-slice (move-type-specialised positions; open-file tactical
  endgames, knight-outpost tactical positions) PR AUC lifts at least
  +0.02 over i193, AND
- A1 (`single_move_type`) loses >= 50% of that lift, AND
- A4 (`shuffle_adjacency`) loses >= 70% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

The DAG source primitive lists ROP (rank-order pooling) as a sibling
primitive. ROP is deferred:

- ROP overlaps with the existing `soft_sorting_order_residual_ranker`
  trunk model and should be evaluated separately.
- IIG (color-involution gate) is an orthogonal symmetry primitive and
  should be evaluated separately.

Run these only after the primary falsifier (`single_move_type`) passes.

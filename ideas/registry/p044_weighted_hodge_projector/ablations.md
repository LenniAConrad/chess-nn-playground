# Ablations

p044 supports eight ablation modes via `model.ablation`. The primary
falsifier is `uniform_metric` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `uniform_metric` | Fix ``W = I``; the decomposition becomes a fixed linear projection. **Primary falsifier.** If A1 matches the unablated run, the input-dependent metric is not load-bearing. |
| A2 | `drop_curl` | Zero the curl branch. Tests whether circulation matters. |
| A3 | `drop_gradient` | Zero the gradient branch. |
| A4 | `drop_harmonic` | Zero the harmonic residual. |
| A5 | `shuffle_edge_flow` | In-batch permutation of the per-edge flow tensor. Decouples flow values from edge geometry. |
| A6 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A7 | `trunk_only` | Same as A6 (semantic alias). |
| A8 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p044 >= i193 - 0.005, AND
- the target-slice ("fortress / circulation / trapped pressure" puzzles
  per the source primitive) PR AUC lifts at least +0.02 over i193, AND
- A1 (`uniform_metric`) loses >= 35% of that lift, AND
- at least one of {A2, A3, A4} loses >= 25% of that lift (otherwise the
  decomposition is uninformative), AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Exact pseudo-inverse decomposition*: replace eps-shift with pinv.
- *Sherman-Morrison metric update*: bounded-change inference.
- *Larger flow_channels*: with the SPD solves shared across channels,
  larger ``flow_channels`` is essentially free; deferred until
  keep-decision.

Run these only after the primary falsifier (`uniform_metric`) passes.

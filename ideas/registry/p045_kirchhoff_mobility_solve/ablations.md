# Ablations

p045 supports seven ablation modes via `model.ablation`. The primary
falsifier is `uniform_conductance` -- every promotion run must include
this matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `uniform_conductance` | Replace the learned positive conductance with all-ones. **Primary falsifier.** If A1 matches the unablated run, the input-dependent conductance is not load-bearing -- the resolvent collapses to a fixed linear map of the source. |
| A2 | `diagonal_only` | Drop the Laplacian term entirely; ``u = s / shift``. Tests whether the resolvent structure matters beyond a per-square source readout. |
| A3 | `shuffle_conductance` | In-batch permutation of the conductance vector. Decouples conductance from board geometry. |
| A4 | `zero_source` | Zero the source term; ``u = 0``. Sanity check. |
| A5 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A6 | `trunk_only` | Same as A5 (semantic alias). |
| A7 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p045 >= i193 - 0.005, AND
- the target-slice ("king-safety / bottleneck / fortress" puzzles per
  the source primitive) PR AUC lifts at least +0.02 over i193, AND
- A1 (`uniform_conductance`) loses >= 35% of that lift, AND
- A2 (`diagonal_only`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Multi-pole source*: include both attacker-zone and king-zone source
  channels with opposite signs.
- *Sparse solver*: replace dense SPD solve with Cholesky on a sparse
  band structure.
- *Conjugate gradient*: when ``output_channels`` grows.

Run these only after the primary falsifier (`uniform_conductance`)
passes.

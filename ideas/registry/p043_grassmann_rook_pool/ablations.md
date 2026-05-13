# Ablations

p043 supports seven ablation modes via `model.ablation`. The primary
falsifier is `drop_exclusion` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `drop_exclusion` | Replace the row/column-disjoint scan with the flat elementary-symmetric pool over the edge tensor. **Primary falsifier.** If A1 matches the unablated run, the rook-matching constraint is not load-bearing. |
| A2 | `scalar_score` | Collapse the H edge-score channels to one (broadcast mean). Tests whether multi-channel edge representation matters. |
| A3 | `shuffle_attackers` | In-batch permutation of attacker tokens. Decouples the attacker side from positions. |
| A4 | `shuffle_defenders` | In-batch permutation of defender tokens. |
| A5 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A6 | `trunk_only` | Same as A5 (semantic alias). |
| A7 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p043 >= i193 - 0.005, AND
- the target-slice ("overloaded defender / two-attacker race"
  puzzles per the source primitive) PR AUC lifts at least +0.02 over
  i193, AND
- A1 (`drop_exclusion`) loses >= 40% of that lift (i.e. row/column
  exclusion is load-bearing), AND
- A2 (`scalar_score`) loses >= 25% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *K=3 rook scan*: extend ``degree`` to 3; already supported but slower.
  Run only after K=2 keep-decision is in.
- *Sherman-Morrison delete update*: bounded-change inference.
- *Sparse top-k edge selection*: drop low-scoring edges before the scan.

Run these only after the primary falsifier (`drop_exclusion`)
passes.

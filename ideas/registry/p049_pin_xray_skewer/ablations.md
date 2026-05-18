# Ablations

p049 supports seven ablation modes via `model.ablation`. The primary
falsifier is `no_xray1` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `no_xray1` | Zero every event term that depends on `second_occ` (one-blocker x-ray). **Primary falsifier.** If A1 matches the unablated run, the operator is not actually using x-ray-through-one-blocker logic and is effectively reduced to clear-ray geometry. |
| A2 | `uniform_values` | Replace the per-piece-type value softmax with a uniform `1/6` field. Tests whether king / queen / rook value context is load-bearing. |
| A3 | `no_pin_def` | Zero the pinned-defender event channel. Tests whether the additional defender-load proxy buys anything. |
| A4 | `shuffle_rays` | In-batch permutation of the `(8, 64, 7)` ray-index table. Decouples the rule-derived ray geometry from the position. |
| A5 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A6 | `trunk_only` | Same as A5 (semantic alias). |
| A7 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p049 >= i193 - 0.005, AND
- at least one of {`pin`, `skewer`, `discovered_attack`} slice PR
  AUC lifts >= +0.01 over i193, AND
- A1 (`no_xray1`) loses >= 50% of that lift (i.e. the x-ray /
  second-occupant logic is load-bearing), AND
- A2 (`uniform_values`) loses >= 30% of that lift, AND
- A4 (`shuffle_rays`) loses essentially all of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than
  the aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Native i018 relation integration* (phase 3 of the source spec).
  This would replace the additive-head fusion with appending the six
  event channels to i018's relation tensor. Out of scope for this
  idea-folder; the source markdown lists it as the next step after a
  positive standalone keep-decision.
- *Full defender-load form*: replace the simple `first_value *
  second_king` proxy with the spec's
  `D_def(b) = sum_u A_same(b, u) v(u)`. Requires building a same-
  side defence graph; defer until the simpler proxy is shown to be
  load-bearing.
- *Third-occupant tier*: extend the cumsum-based occupant masks to
  track the third occupied square as well, enabling longer-range
  latent threats (e.g. queen behind two friendly defenders behind
  enemy king).

Run these only after the primary falsifier (`no_xray1`) passes.

# Ablations

p050 supports seven ablation modes via `model.ablation`. The primary
falsifier is `no_cross_target_load` -- every promotion run must
include this matched control on the same split, seed, and training
budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `no_cross_target_load` | Replace `L^2 - Σ_t O^2` with `Σ_t O^2`. Strips out the cross-target overload mass, leaving only single-target under-defence. **Primary falsifier.** If A1 matches the unablated run, the operator is not actually using defender-identity reuse. |
| A2 | `no_pins` | Set `π = 0` everywhere. Tests whether pinned defenders contribute on top of plain attack/defence. |
| A3 | `no_target_value` | Set `v_tar = 1` on every occupied target (and `v_att, v_def = 1`). Tests whether piece-value weighting matters. |
| A4 | `counts_only` | Drop `a_val, d_val, m_att, m_def` from the target-criticality MLP, leaving only counts and `v_tar`. Tests SEE-light feature load-bearingness. |
| A5 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193 numerically. |
| A6 | `trunk_only` | Same as A5 (semantic alias). |
| A7 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p050 >= i193 - 0.005, AND
- at least one of {`overload`, `pin`, `deflection`} slice PR AUC
  lifts >= +0.01 over i193, AND
- A1 (`no_cross_target_load`) loses >= 50% of that lift (i.e. the
  defender-identity-reuse term is load-bearing), AND
- A2 (`no_pins`) loses >= 20% of that lift on the `pin` slice, AND
- A3 (`no_target_value`) loses >= 30% of that lift on at least the
  `overload` slice, AND
- the `crtk_eval_bucket = equal` slice does not regress more than
  the aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Native i018 relation integration* (phase 3 of the source spec).
  This would replace the additive-head fusion with appending the
  defender-overload pool to i018's relation tensor. Out of scope
  for this idea-folder; the source markdown lists it as the next
  step after a positive standalone keep-decision.
- *Full SEE recursive exchange evaluation*: replace the cheapest-
  attacker / cheapest-defender minima with the full Stockfish-style
  recursive Static Exchange Evaluation. Requires building an
  exchange stack; defer until the cheap proxies are shown to be
  load-bearing.
- *Empty king-ring overload pass*: optional second `_side_stats`
  invocation over empty king-ring targets with a surrogate
  `v_ring` constant, to extend the operator to mate-square overload.
  Currently out of scope; the spec lists it as an optional second
  variant.
- *Defender-shuffle falsifier*: shuffle defender rows per target
  within value × pin buckets (the spec's structure-destroying
  falsifier). Requires a tagged-bucket scrambler; defer to the
  next iteration.

Run these only after the primary falsifier
(`no_cross_target_load`) passes.

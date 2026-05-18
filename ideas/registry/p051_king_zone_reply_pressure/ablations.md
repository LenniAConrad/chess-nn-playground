# Ablations

p051 supports nine ablation modes via `model.ablation`. The primary
falsifier is `no_front_zone` -- every promotion run must include
this matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `no_front_zone` | Drop the forward-rank zone term `η · Σ_{q ∈ Z_front} net`. Strips out the front-rank extension that distinguishes KZRP from a plain ring-only king-pressure scalar. **Primary falsifier.** If A1 matches the unablated run, the front-rank extension is not load-bearing and the operator collapses to ring-only. |
| A2 | `no_pins` | Set `π = 0` everywhere; collapse `D_free = D_nom`. Tests whether the nominal-vs-free defender distinction is load-bearing. |
| A3 | `uniform_zone_weights` | Replace `(king_sq=4, empty_ring=3, occupied_ring=2)` zone weights with uniform 1. Tests whether unequal weighting of king-zone squares matters. |
| A4 | `no_escape_decomp` | Collapse the three escape-class counts (live, sealed, blocked) into a single total. Tests whether the decomposition is load-bearing or whether a single mobility scalar suffices. |
| A5 | `uniform_units` | Set the attack-unit field `u = 1` uniformly. Tests CPW-style `(P=1, N=2, B=2, R=3, Q=5)` weighting. |
| A6 | `no_asymmetry` | Zero `S_them`. Tests whether the side-to-move asymmetry `S_us - S_them` is load-bearing. |
| A7 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193 numerically. |
| A8 | `trunk_only` | Same as A7 (semantic alias). |
| A9 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p051 >= i193 - 0.005, AND
- at least one of {`mate_in_1`, near-puzzle FP at recall 0.8,
  `discovered_attack`} lifts >= +0.005 over i193, AND
- A1 (`no_front_zone`) loses >= 30% of that lift on at least
  `mate_in_1` (the front-zone term is supposed to be the
  load-bearing extension over a ring-only baseline), AND
- A2 (`no_pins`) loses >= 20% of the lift on positions where the
  pin slice intersects king-zone activity, AND
- A6 (`no_asymmetry`) loses >= 30% of the lift on `mate_in_1` (the
  side-to-move asymmetry is supposed to be the primary discriminator
  between forcing-attack and mutual-king-danger positions), AND
- the `crtk_eval_bucket = equal` slice does not regress more than
  the aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Native i018 `TacticalReadout` integration* (phase 2 of the
  source spec). This would replace the additive-head fusion with
  appending the raw KZRP vector terms to the i018 `TacticalReadout`.
  Out of scope for this idea-folder; the source markdown lists it
  as the next step after a positive standalone keep-decision.
- *BT4 plane augmentation* (phase 3 of the source spec). Add the
  4 spatial maps as extra BT4 stem channels and inject the global
  vector into the value head. Requires a different host model and
  is left for a follow-up idea.
- *BT4 mixer-native variant* (phase 3 of the source spec). Build a
  `bt4_mixers/king_zone_reply_pressure.py` mixer under the
  `bt4_primitive_mixer` harness. Requires the BT4 mixer scaffold and
  is the highest-risk phase per the source spec.
- *Full check-evasion reply enumerator*: replace the cheap
  `reply_proxy` with an exact king-move / capture / interposition
  count. Requires building a check-evasion family generator; defer
  until the cheap proxy is shown to be load-bearing on `mate_in_1`.
- *Spatial output maps*: the source spec also calls for emitting
  4 spatial maps (`zone_weight_map`, `net_control_map`,
  `escape_state_map`, `pin_line_map`). The current implementation
  only exposes the global vector and per-side scalars; spatial maps
  are deferred to the BT4 plane-augmentation variant where they
  would actually be consumed by a spatial trunk.

Run these only after the primary falsifier (`no_front_zone`)
passes.

# Ablations

p054 supports nine ablation modes via `model.ablation`. The primary
falsifier is `first_only`; every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `first_only` | Zero everything beyond the first blocker (no second-blocker channels, no x-ray pressure, no discovered / pin candidates). **Primary falsifier.** If A1 matches the unablated run, the second-blocker structure (and its derived x-ray / discovered / pin signals) is not load-bearing. |
| A2 | `no_blocker_id` | Zero side / value identity channels for both first and second blockers. Keeps visibility / mobility / x-ray geometry. Tests whether *what* sits at the blocker matters versus pure geometry. |
| A3 | `uniform_occupancy` | All squares treated as occupied. Only the first ray cell is ever visible. Tests whether the chess-rule occupancy mask carries signal. |
| A4 | `empty_occupancy` | Empty board: pure geometric ray length. Tests whether the head can still discriminate on positions when only mobility geometry is left. |
| A5 | `shuffle_occupancy` | In-batch permutation of the occupancy mask. Decouples mask from position. |
| A6 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A7 | `trunk_only` | Same as A6 (semantic alias). |
| A8 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p054 >= i193 - 0.005, AND
- the target-slice ("second-blocker-dependent" tactical puzzles, e.g.
  x-ray attacks, discovered-attack frames, soft pins) PR AUC lifts at
  least +0.02 over i193, AND
- A1 (`first_only`) loses >= 50% of that lift, AND
- A2 (`no_blocker_id`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Dense-edge scatter mode* (`rook_visible`, `bishop_visible`,
  `rook_xray`, `bishop_xray`, `queen_*`) per the source markdown's
  PyTorch pseudocode. Promote behind a future i018 graph-builder
  integration if the compact head proves predictive.
- *Builder-replacement* mode where p054 swaps the i018 visibility
  kernel directly. Requires the dense-edge scatter mode and an
  audit-equivalence test against `TacticalIncidenceBuilder`.
- *`torch.compile` benchmark* of the compact scan against p020 / p021
  / p026 and the i018 dense visibility builder. The scan body is
  static-shape and a natural fit for `torch.compile`; the realised
  speedup depends on `gather + cumsum` fusion on the target GPU and
  must be measured before the "efficient" half of the primitive
  thesis can be claimed.

Run these only after the primary falsifier (`first_only`) passes.

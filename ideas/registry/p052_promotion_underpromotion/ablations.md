# Ablations

p052 supports eight ablation modes via `model.ablation`. The primary
geometry falsifier is `pseudo_only` -- every promotion run must
include this matched control on the same split, seed, and training
budget. The underpromotion-hint, capture-geometry, arrival-safety,
and gate ablations each test a distinct structural claim.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `pseudo_only` | **Primary falsifier**. Drop legality filtering: candidates fire on every own near-promotion pawn regardless of arrival-square occupancy. If A1 matches the unablated run, the exact promotion geometry (legality filtering of push and capture candidates) is not load-bearing. |
| A2 | `no_capture` | Drop diagonal capture promotion candidates. If A2 matches `none`, the capture-promotion geometry is not load-bearing and the primitive can be simplified to push-only. |
| A3 | `queen_only` | Zero out rook / bishop / knight delta-to-queen channels and the knight-fork hint `kappa_N`. If A3 matches `none`, the underpromotion-hint story is false and PUGP should be re-framed as a promotion-only primitive. |
| A4 | `no_attack_defense` | Zero out arrival-square attack/defense counts, gives-check, king-zone overlap and the knight-fork hint. If A4 matches `none`, the arrival-square safety/check story is not doing the work claimed for it. |
| A5 | `zero_delta` | Zero primitive delta. Recovers the i193 baseline. |
| A6 | `trunk_only` | Same as A5 (semantic alias). |
| A7 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing -- if A7 matches `none`, the gate is not actually filtering the delta into a no-op on non-promotion samples. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p052 >= i193 - 0.005, AND
- merged ``promotion / underpromotion`` slice PR AUC lifts at least
  +0.02 over i193, AND
- A1 (`pseudo_only`) loses >= 50% of that lift, AND
- (A2 or A3 or A4) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Source-pawn-removal correction*: subtract the source pawn's
  blocker contribution on capture-promotion southward-diagonal rays
  before counting sliding attackers. Currently unimplemented (small
  conservative bias; documented in `implementation_notes.md`).
- *Precompute path*: a Parquet column variant of PUGP features. The
  ablation would be "compute in-forward" vs "load precomputed",
  designed to verify they produce numerically identical features.
- *BT4 mixer study*: PUGP wired into the `bt4_primitive_mixer` shape,
  analogous to `a003_bt4_promotion_aware_head_mixer`. Deferred until
  after this primitive proves a keep decision as an additive side head.

Run these only after the primary falsifier (`pseudo_only`) passes.

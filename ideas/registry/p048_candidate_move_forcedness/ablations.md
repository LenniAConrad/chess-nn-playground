# Ablations

p048 supports eight ablation modes via `model.ablation`. The primary
falsifier is `deterministic_score` -- every promotion run must
include this matched control on the same split, seed, and training
budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `deterministic_score` | Replace per-move learned score with the deterministic feature sum. **Primary falsifier.** If A1 matches the unablated run, the learned scorer is not load-bearing. |
| A2 | `mean_pool` | Replace top-k pooling with mean over all legal candidates. Tests whether candidate concentration matters. |
| A3 | `flags_only` | Keep only move-class flags; drop piece values, mobility, and SEE-lite. Tests whether deeper features earn their cost. |
| A4 | `dense_edges` | Replace pseudo-legal adjacency with a fully-connected mask. Tests whether exact legality / geometry matters beyond all-pairs. |
| A5 | `no_consequence` | Drop check / capture / promotion seeds and SEE-lite. Tests whether forcing-class features matter. |
| A6 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A7 | `trunk_only` | Same as A6 (semantic alias). |
| A8 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p048 >= i193 - 0.005, AND
- the target-slice ("forcing-line tactics" per the source primitive)
  PR AUC lifts at least +0.02 over i193, AND
- A1 (`deterministic_score`) loses >= 50% of that lift, AND
- A2 (`mean_pool`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than
  the aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Post-move board apply*: extend the edge features with
  `threat_creation`, `escape_reduction`, and `evasion_scarcity`
  from a true post-move pass. Deferred behind the dense pilot.
- *Candidate-major compaction*: replace the dense `(B, 64, 64)`
  edge tensor with a packed candidate-major layout. Deferred until
  memory becomes the binding constraint.
- *Castling / en-passant edges*: extend
  `compute_legal_move_graph` with castling and en-passant edge
  emissions. Deferred behind the legality keep-decision.

Run these only after the primary falsifier
(`deterministic_score`) passes.

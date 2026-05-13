# Ablations

p035 supports seven ablation modes via `model.ablation`. The primary
falsifier is `separable_phi` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `separable_phi` | Zero the Hadamard interaction term. **Primary falsifier.** If A1 matches the unablated run, the joint edge function is not load-bearing -- the operator collapses to a separable GAT-style aggregation. |
| A2 | `uniform_adjacency` | Replace ``A`` with all-ones (minus identity). Tests whether the chess-rule mask matters. |
| A3 | `shuffle_adjacency` | In-batch permutation of the legal-move graph. Decouples rule indicators from positions. |
| A4 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A5 | `trunk_only` | Same as A4 (semantic alias). |
| A6 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p035 >= i193 - 0.005, AND
- the target-slice ("hanging piece" puzzles per the source primitive)
  PR AUC lifts at least +0.02 over i193, AND
- A1 (`separable_phi`) loses >= 40% of that lift, AND
- A2 (`uniform_adjacency`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Per-piece-type edge function*: condition ``phi`` on the piece type
  at source and target.
- *Sparse-edge formulation*: replace the (B, 64, 64, d_edge) pair
  tensor with an explicit |E|-edge tensor + index_add_; needed for
  larger batches.
- *Two-hop aggregation*: apply ``phi`` twice with the same adjacency;
  captures 2-ply tactical reasoning.

Run these only after the primary falsifier (`separable_phi`) passes.

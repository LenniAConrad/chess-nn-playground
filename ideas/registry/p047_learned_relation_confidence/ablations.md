# Ablations

p047 supports nine ablation modes via `model.ablation`. The primary
falsifier is `binary_only` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `binary_only` | Skip per-edge confidence; summary inputs reduce to raw deterministic mask statistics. **Primary falsifier.** If A1 matches the unablated run, the confidence layer is not load-bearing. |
| A2 | `scrambled_mask` | In-batch permute the deterministic relation masks. Tests whether mask-position alignment carries signal. |
| A3 | `shuffle_pieces` | In-batch permute the per-square piece descriptor. Tests whether piece identity matters in the edge MLP. |
| A4 | `gate_only` | Disable per-edge scoring; only the per-relation gate is learned. Tests whether per-edge structure beats coarse rescaling. |
| A5 | `no_low_rank` | Drop the low-rank bilinear term. |
| A6 | `no_edge_mlp` | Drop the edge MLP; keep low-rank only. |
| A7 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A8 | `trunk_only` | Same as A7 (semantic alias). |
| A9 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p047 >= i193 - 0.005, AND
- the target-slice ("strength-of-known-relation" puzzles per the
  source primitive) PR AUC lifts at least +0.02 over i193, AND
- A1 (`binary_only`) loses >= 50% of that lift, AND
- A4 (`gate_only`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *Sparse downstream consumption*: rowwise differentiable top-`k`
  pruning followed by sparse gather/scatter for the downstream
  consumer. Deferred behind the dense keep-decision.
- *Graph-side adapter*: replace `exchange_soundness_graph_network`
  attack/defense intensity heuristics with the LRC weighted attack /
  defense edges. Run only after the primary falsifier (`binary_only`)
  passes.

Run these only after the primary falsifier (`binary_only`) passes.

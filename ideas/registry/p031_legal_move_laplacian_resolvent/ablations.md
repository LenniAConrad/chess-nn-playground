# Ablations

p031 supports eight ablation modes via `model.ablation`. The primary
falsifier is `k1_gat_rebrand` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `k1_gat_rebrand` | Force ``K=1``. Collapses the Neumann series to a single hop, reducing the operator to GAT-with-legal-mask. **Primary falsifier.** If A1 matches the unablated run on the target slices, the Neumann expansion is not load-bearing and the primitive is dropped. |
| A2 | `uniform_piece_weights` | Set per-piece edge weights to 1. Tests whether piece-conditioned weighting is load-bearing. |
| A3 | `shuffle_adjacency` | Permute the legal-move graph across the batch. Decouples the rule indicators from the position. |
| A4 | `zero_alpha` | Force ``alpha = 0`` -> ``Y = X * Theta``. Tests whether the resolvent expansion is load-bearing at all. |
| A5 | `zero_delta` | Zero primitive delta. Recovers i193 trunk behavior. |
| A6 | `trunk_only` | Same as A5 (semantic alias for the strongest control). |
| A7 | `disable_gate` | Pin the gate at 1.0. Tests whether the learned gate is load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p031 >= i193 - 0.005, AND
- the primitive's declared target slice (multi-hop tactical / hard-negative
  near-puzzle) PR AUC lifts at least +0.02 over i193, AND
- A1 (`k1_gat_rebrand`) loses >= 50% of the target-slice lift, AND
- A3 (`shuffle_adjacency`) loses >= 70% of the target-slice lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails. Drop especially if A1 matches the unablated
run -- that means the head learned to use a single-hop legal-mask GAT
without relying on the resolvent expansion at all.

## Out-of-scope ablations (future)

The LM-LPP source primitive lists three further ablations that need new
architecture surface area and are out of scope for the first scout run:

- *Spectral clipping of alpha by power-iteration*: required to enable
  larger ``K`` without numerical drift; tracked as an implementation
  upgrade in ``implementation_notes.md``.
- *Sparse CSR matmul kernel*: required to capture the asymptotic
  complexity win on consumer GPUs.
- *Triton kernel for fused Neumann expansion*: would fuse the ``K``
  matmuls into a single launch.

Run these only after the primary falsifier (`k1_gat_rebrand`) passes.

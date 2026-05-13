# Ablations

p004 supports eight ablation modes via `model.ablation`. The primary
falsifier is `rank_quantile_only` — the main novelty claim is "tail
concordance beats marginal rank quantiles."

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `rank_quantile_only` | Collapse the concordance matrix to identity; only marginal rank-quantile information remains. **Primary falsifier vs `i095`-style baselines.** |
| A2 | `square_shuffle` | Shuffle squares per channel; destroys cross-site alignment. Should kill most of the lift. |
| A3 | `channel_shuffle` | Permute channels; destroys cross-channel concordance. |
| A4 | `single_channel` | Use only channel 0; concordance is trivial. |
| A5 | `disable_gate` | Hold `primitive_gate` at 1.0. |
| A6 | `zero_delta` | Zero out `primitive_delta`. Recovers i193 baseline. |
| A7 | `trunk_only` | Zero out features and delta. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p004 >= i193 - 0.005, AND
- matched-recall near-puzzle FP at recall 0.80 improves by at least
  2% versus `i095`-style baselines on the same parent, AND
- A1 (`rank_quantile_only`) loses >= 50% of the near-FP lift, AND
- A2 (`square_shuffle`) loses >= 70% of the lift (proves cross-site
  alignment matters).

Drop especially if A1 matches the unablated run — that means the
marginal rank quantiles were sufficient and the cross-channel
concordance is not load-bearing.

## Out-of-scope ablations (future)

- Replace the `O(N^2 * C)` pairwise soft-rank with a fused
  differentiable sorting kernel.
- Combine TCC with PAFR by using the tail-hotspot map as a candidate
  attention mask.
- Add a per-region tail mass pooled around king zones / promotion
  lanes.

Run these only after the primary falsifier (`rank_quantile_only`)
passes.

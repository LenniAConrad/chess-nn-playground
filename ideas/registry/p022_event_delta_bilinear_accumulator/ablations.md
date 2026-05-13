# Ablations

p022 supports four ablation modes via `model.ablation`. The primary
falsifier is `first_order_only` -- it removes the bilinear pair term.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `first_order_only` | Drop the pair term `Q` (set to 0). **The primary falsifier.** If A1 matches `none` on the declared slice, the bilinear term is not load-bearing and the head collapses to first-order accumulation. |
| A2 | `shuffle_pair_term` | In-batch permutation of `Q`. Decouples the pair signal from positions. Tests whether the trunk + first-order term alone explain the lift. |
| A3 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A4 | `trunk_only` | Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p022 >= i193 - 0.005, AND
- declared interaction slice PR AUC of unablated p022 >= i193 + 0.04, AND
- A1 (`first_order_only`) loses >= 70% of the slice lift, AND
- training throughput drop versus i193 < 25%.

Drop especially if A1 matches `none` -- the pair-term claim is the
defining novelty.

## Out-of-scope ablations (future)

- Replace the FM identity with explicit pair enumeration to confirm
  numerical agreement.
- Vary `bilinear_dim` to map the cost / accuracy frontier.
- Disable normalisation by active count.

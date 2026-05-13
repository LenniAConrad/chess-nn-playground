# Ablations

p024 supports five ablation modes via `model.ablation`. The primary
falsifier is `first_order_only` -- it disables the higher-order
elementary symmetric states.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `first_order_only` | Zero out `E^{(>=2)}`; keep only `E^{(1)}`. **The primary falsifier.** Collapses to EmbeddingBag-style sum. |
| A2 | `second_order_only` | Zero out `E^{(1)}` and `E^{(3)}`; keep only `E^{(2)}`. Tests the second-order term alone. |
| A3 | `shuffle_higher_orders` | In-batch permutation of `E^{(>=2)}`. Decouples higher orders from positions. |
| A4 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A5 | `trunk_only` | Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p024 >= i193 - 0.005, AND
- declared higher-order slice PR AUC of unablated p024 >= i193 + 0.04, AND
- A1 (`first_order_only`) loses >= 70% of the slice lift, AND
- training throughput drop versus i193 < 25%.

Drop especially if A1 matches `none` -- the higher-order interaction
claim is the defining novelty.

## Out-of-scope ablations (future)

- Vary `order` (R = 2 vs R = 3).
- Replace the Hadamard recurrence with Newton's identities on power
  sums (verify numerical agreement).
- Disable normalisation by active count.

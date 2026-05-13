# Ablations

p023 supports five ablation modes via `model.ablation`. The primary
falsifier is `disable_bilinear` -- it removes the bilinear hyperedge
contraction (the file-name promise of the primitive).

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `disable_bilinear` | Replace `left * right` with `left + right`. **The primary falsifier.** If A1 matches `none` on the declared slice, the bilinear hyperedge claim fails. |
| A2 | `zero_occupancy` | Treat the board as empty. Removes the transmittance gate. |
| A3 | `uniform_occupancy` | Full blocker everywhere. Recurrence carries nothing. |
| A4 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A5 | `trunk_only` | Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p023 >= i193 - 0.005, AND
- declared through-the-square slice PR AUC of unablated p023 >=
  i193 + 0.04, AND
- A1 (`disable_bilinear`) loses >= 70% of the slice lift, AND
- training throughput drop versus i193 < 25%.

Drop especially if A1 matches `none`.

## Out-of-scope ablations (future)

- Use a single shared `W_LR` matrix for left and right projections.
- Vary the number of opposing-direction pairs.
- Replace the bilinear contraction with a low-rank Kronecker product.

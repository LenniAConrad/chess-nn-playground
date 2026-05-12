# Implementation Notes

## Implementation Plan

1. Implement deterministic tactical beam generation.
2. Encode move descriptors without engine scores.
3. Implement latent transition over move descriptors.
4. Implement differentiable proof-number aggregation.
5. Register `neural_proof_number_search`.
6. Run depth-1, depth-2, and depth-3 ablations.

## Tactical Beam Ordering

The beam may use deterministic rule-only priority:

- checks
- captures
- promotions
- moves near opponent king
- attacks on high-value pieces
- line-opening moves
- defender captures or blocks for AND replies

Do not use Stockfish, tablebases, or generated best-move labels.

## Solver/Aggregator Choice

Use temperature-controlled functions:

```text
softmin(x) = -tau * logsumexp(-x / tau)
softsum(x) = tau * logsumexp(x / tau)
```

This gives stable gradients and represents proof/disproof number asymmetry.

## Known Risks

- Move tree generation may be slow.
- Binary supervision may not teach good proof costs.
- Beam pruning may miss the key move.
- The model may learn move-count or check-count shortcuts.
- Depth-3 may be too expensive without careful batching.

## Testing Plan

- Unit test move tree shapes for simple FENs.
- Unit test AND/OR aggregation math on synthetic costs.
- Unit test forward output and diagnostic shapes.
- Tiny depth-1 smoke train before depth-3 benchmark.


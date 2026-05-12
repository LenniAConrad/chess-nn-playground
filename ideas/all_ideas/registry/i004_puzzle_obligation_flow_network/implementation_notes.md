# Implementation Notes

## Implementation Plan

1. Implement deterministic obligation/resource candidate extraction.
2. Build padded tensors and compatibility masks.
3. Implement a simple dense unrolled allocation solver first.
4. Implement `PuzzleObligationFlowNetwork`.
5. Register the model under `puzzle_obligation_flow_network`.
6. Add one small smoke config before full benchmark training.

## Minimal Candidate Generator

Start small:

- king-neighborhood obligations
- attacked high-value piece obligations
- slider-line block/capture obligations
- defender resources from pieces attacking obligation squares
- king moves and interpositions where legal generation is available

Do not try to enumerate every chess concept in the first version.

## Solver Choice

First version can use a soft unrolled primal-dual update:

```text
P = sigmoid(compatibility_logits)
P = project rows toward demand
P = project columns toward capacity
repeat solver_steps
```

Exact optimal transport is not required initially. The falsifiable point is whether constrained allocation residuals help.

## Dependencies

Use PyTorch. Use existing chess/FEN tooling in the repo. Avoid adding a heavy optimization dependency.

## Known Risks

- Candidate generation may dominate implementation complexity.
- The learned flow may be too soft and behave like attention.
- A shallow candidate set may miss important tactical obligations.
- Model may overfocus on king attacks and miss material puzzles.

## Testing Plan

- Unit test candidate tensor shapes from simple FENs.
- Unit test compatibility masks are deterministic.
- Unit test allocation output satisfies rough row/column constraints.
- Unit test forward output shape.
- Tiny training smoke test on `puzzle_binary`.


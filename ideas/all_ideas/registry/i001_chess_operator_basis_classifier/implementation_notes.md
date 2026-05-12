# Implementation Notes

## Implementation Plan

1. Add a relation-operator builder that returns sparse or dense 64x64 masks.
2. Implement a `ChessOperatorBasisBlock`.
3. Register `chess_operator_basis_classifier` in the model registry.
4. Add one `simple_18` benchmark config with `mode: puzzle_binary` and `num_classes: 1`.
5. Run against MLP, CNN, NNUE, and BT4 baselines.

## Dependencies

Use PyTorch only. The first version can store operators as dense buffers because 64x64 is small. Convert to sparse only if profiling justifies it.

## Known Risks

- Too many operators may make the model a noisy superset of a CNN.
- Occupancy-gated line operators may accidentally duplicate source artifacts.
- If the operator gate dominates, ablations must verify that each relation family matters.

## Testing Plan

- Unit test output shape for `simple_18`.
- Unit test deterministic operator masks.
- Unit test CPU forward pass.
- Train a 1-epoch smoke run on a tiny split.


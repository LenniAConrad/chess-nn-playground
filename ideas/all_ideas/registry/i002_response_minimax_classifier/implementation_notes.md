# Implementation Notes

## Implementation Plan

1. Implement deterministic pseudo-legal action extraction using the existing chess dependency if available.
2. Add capped deterministic ordering for actions and replies.
3. Implement token encoders that gather board features at `from` and `to` squares.
4. Implement soft minimax pooling.
5. Register the model only after CPU forward tests pass.

## Dependencies

Prefer `python-chess` if already available in the environment. If not, add it explicitly to requirements only after checking the repo dependency policy.

## Known Risks

- Move generation may become the training bottleneck.
- Capping moves/replies may drop decisive moves.
- If labels are static and not action-response-driven, the model may add noise.

## Testing Plan

- Unit test deterministic move extraction for simple FENs.
- Unit test padded action/reply tensor shapes.
- Unit test permutation invariance after move ordering is fixed.
- Tiny smoke training run.


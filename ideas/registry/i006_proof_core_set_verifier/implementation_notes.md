# Implementation Notes

## Implementation Plan

1. Implement candidate token extraction.
2. Implement deterministic relation features among candidate tokens.
3. Implement soft top-k selection first, then optionally hard straight-through top-k.
4. Implement relation-aware set verifier.
5. Add deletion diagnostic pass for validation/test only.

## Dependencies

Use PyTorch only.

## Known Risks

- Top-k selection may be unstable early in training.
- The selector may collapse to king squares only.
- The bounded residual may still leak too much global information.
- Small cores may miss distributed tactical themes.

## Testing Plan

- Unit test token extraction shape.
- Unit test relation tensor symmetry/asymmetry where expected.
- Unit test selector returns exactly `k` witnesses in hard mode.
- Unit test forward output and witness diagnostics.
- Tiny smoke training run.


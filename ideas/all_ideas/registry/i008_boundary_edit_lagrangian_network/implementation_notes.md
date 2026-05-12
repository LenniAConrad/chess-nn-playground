# Implementation Notes

## Implementation Plan

1. Implement edit-basis generation over latent deltas.
2. Implement an unrolled projected-gradient or Lagrangian solver over edit weights.
3. Compute both `E_plus` and `E_minus`.
4. Add diagnostics to prediction artifacts.
5. Register `boundary_edit_lagrangian_network`.

## First Edit Basis

Start with a small basis:

- side-to-move contrast
- defender weakening around attacked targets
- slider blocker removal/opening
- king escape square suppression
- target protection reduction
- pinned-piece relation toggle

Keep edits soft in latent space. Do not emit edited boards as labeled training data.

## Solver Choice

First implementation:

```text
alpha = sigmoid(initial_edit_logits)
for step in solver_steps:
    compute energy
    gradient update on alpha
    project alpha into [0, 1]
```

This can be implemented as unrolled differentiable computation.

## Known Risks

- The solver may collapse to always choosing the same edit type.
- The final head may ignore edit energies.
- The edit basis may be too narrow.
- Energies may be poorly calibrated without auxiliary regularization.

## Testing Plan

- Unit test edit-basis tensor shapes.
- Unit test solver keeps alpha in `[0, 1]`.
- Unit test forward diagnostics.
- Smoke train against a tiny puzzle-binary split.


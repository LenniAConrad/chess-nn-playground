# Implementation Notes

## Implementation Plan

1. Add a deterministic null-view tensor transform.
2. Implement a shared CNN or operator-basis encoder.
3. Add evidence heads for current and null views.
4. Add a pair mixer and final puzzle head.
5. Log current/null evidence diagnostics.

## Dependencies

Use PyTorch only. Reuse existing feature encoding where possible.

## Known Risks

- Side-to-move swap may be too crude for positions in check.
- The model may learn a side-to-move shortcut.
- Some puzzle types may not be tempo-sensitive.

## Testing Plan

- Unit test null-view transform is deterministic.
- Unit test current/null shapes match.
- Unit test forward output and diagnostic shapes.
- Tiny puzzle-binary smoke run.


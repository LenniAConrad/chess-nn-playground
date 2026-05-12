# Implementation Notes

## Implementation Plan

1. Implement the four branches with small dimensions.
2. Initialize the residual joint head near zero.
3. Register `factor_agreement_classifier`.
4. Log factor logits and disagreement statistics to prediction artifacts.
5. Compare against plain concatenation fusion with the same branches.

## Dependencies

PyTorch only.

## Known Risks

- Branches may collapse to the same representation.
- The disagreement penalty may hurt recall.
- A too-strong residual head may bypass agreement.

## Testing Plan

- Unit test factor logit shapes.
- Unit test final logit shape.
- Unit test disabling each branch.
- Tiny smoke training run.


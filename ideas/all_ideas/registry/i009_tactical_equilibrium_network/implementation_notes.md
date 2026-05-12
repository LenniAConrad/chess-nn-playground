# Implementation Notes

## Implementation Plan

1. Implement attacker and defender candidate extraction.
2. Build relation features between candidate pairs.
3. Implement payoff matrix construction.
4. Implement unrolled entropy-regularized game solver.
5. Add diagnostics to prediction artifacts.
6. Register `tactical_equilibrium_network`.

## Candidate Generator

Start with deterministic rule-only candidates:

Attackers:

- checking moves or checking lines
- captures
- attacks on king-zone squares
- attacks on queen/rook/minor targets
- promotion threats
- slider line openings

Defenders:

- king moves
- captures of attacking pieces
- interpositions
- recaptures
- target reinforcement
- counter-threat moves

No engine ordering is allowed.

## Known Risks

- Candidate generator quality may dominate results.
- The matrix game may collapse to max threat score if defender candidates are weak.
- Binary labels may not train interpretable payoffs.
- Long tactics may require proof-number search instead.

## Testing Plan

- Unit test candidate tensor shapes.
- Unit test solver on known synthetic payoff matrices.
- Unit test permutation invariance under candidate reordering.
- Unit test forward output and diagnostics.
- Tiny puzzle-binary smoke run.


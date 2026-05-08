# Math Thesis

Disproof-Ledger Puzzle Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `5`.

Working thesis: The model should not only collect evidence for "puzzle." It should also collect explicit *disproof* evidence — typed reasons the position is sharp but not a puzzle:

- the king can escape
- a defender can recapture
- the line is blocked
- the threat is too slow
- the target is protected enough
- the side to move lacks tempo

Near-puzzles satisfy "this looks tactical" but fail one of these. The model exposes a typed disproof channel for each, computes
`disproof_strength = sum_d softplus(disproof_entry_d)`, and emits

```
puzzle_logit = positive_evidence - disproof_strength
```

so the readout is literally "evidence minus disproof." A small L1 sparsity on `softplus(disproof_entries)` keeps one or two channels dominant, and an optional near-puzzle auxiliary asks that near-puzzle source labels light at least one disproof channel.

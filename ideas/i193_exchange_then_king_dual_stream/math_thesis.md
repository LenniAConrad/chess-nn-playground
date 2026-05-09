# Math Thesis

Exchange-Then-King Dual Stream

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `8`.

Working thesis: Puzzle data likely mixes at least two broad families:

```text
material-winning tactics
king-safety or mate tactics
```

A single trunk may blur the difference. A dual-stream model can let
one branch specialise in material exchange (piece / value /
attacker / defender features) while the other specialises in king
danger (king-zone / escape / check / line features), then a learned
phase router combines the two via

```text
gate = sigmoid(phase_router(board_context))
puzzle_logit = gate * king_logit + (1 - gate) * exchange_logit + residual_logit.
```

The bespoke implementation realises this dual-stream factorisation
on the `simple_18` current-board tensor and feeds the puzzle_binary
BCE-with-logits trainer; see `architecture.md` for the concrete
mechanism, output contract, and ablations.

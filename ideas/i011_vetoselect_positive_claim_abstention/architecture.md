# Architecture

The first implementation intentionally stays close to the strongest current baseline so the change tests the VetoSelect head and loss rather than unrelated representation changes.

- Input: `lc0_bt4_112` current-FEN board tensor.
- Trunk: LC0 BT4-style residual tower with squeeze-excite blocks.
- Head: value-style projection to a hidden vector.
- Evidence branch: `Linear(hidden, 1)` producing `puzzle_logit`.
- Selector branch: `Linear(hidden, 1)` producing `selector_logit`.
- Derived diagnostics: log probabilities for ordinary non-puzzle, rejected evidence, accepted puzzle, plus the selective puzzle logit.

The model does not consume engine columns, best moves, source labels, verification fields, or split metadata.

## Implementation Binding

- Registered model name: `vetoselect_positive_claim_abstention`.
- Source implementation: `src/chess_nn_playground/models/vetoselect.py`.
- Idea-local wrapper: `ideas/i011_vetoselect_positive_claim_abstention/model.py`.

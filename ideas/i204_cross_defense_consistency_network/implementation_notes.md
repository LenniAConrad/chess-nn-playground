# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/cross_defense_consistency.py`.
- Idea-local wrapper: `ideas/i204_cross_defense_consistency_network/model.py` calls the registered builder.
- Registry key: `cross_defense_consistency_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

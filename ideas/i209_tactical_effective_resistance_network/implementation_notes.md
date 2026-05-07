# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/tactical_effective_resistance.py`.
- Idea-local wrapper: `ideas/i209_tactical_effective_resistance_network/model.py` calls the registered builder.
- Registry key: `tactical_effective_resistance_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

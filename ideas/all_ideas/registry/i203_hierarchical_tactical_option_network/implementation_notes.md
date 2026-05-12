# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/hierarchical_tactical_option.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i203_hierarchical_tactical_option_network/model.py` calls the registered builder.
- Registry key: `hierarchical_tactical_option_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

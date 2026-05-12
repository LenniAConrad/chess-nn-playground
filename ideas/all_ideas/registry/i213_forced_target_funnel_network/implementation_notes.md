# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/forced_target_funnel.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i213_forced_target_funnel_network/model.py` calls the registered builder.
- Registry key: `forced_target_funnel_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

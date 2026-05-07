# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/role_counterfactual_necessity.py`.
- Idea-local wrapper: `ideas/i211_role_counterfactual_necessity_network/model.py` calls the registered builder.
- Registry key: `role_counterfactual_necessity_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

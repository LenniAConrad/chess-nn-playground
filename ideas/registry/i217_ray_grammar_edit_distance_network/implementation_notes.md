# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/ray_grammar_edit_distance.py`.
- Idea-local wrapper: `ideas/registry/i217_ray_grammar_edit_distance_network/model.py` calls the registered builder.
- Registry key: `ray_grammar_edit_distance_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/pinned_mobility_nullspace.py`.
- Idea-local wrapper: `ideas/registry/i208_pinned_mobility_nullspace_network/model.py` calls the registered builder.
- Registry key: `pinned_mobility_nullspace_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

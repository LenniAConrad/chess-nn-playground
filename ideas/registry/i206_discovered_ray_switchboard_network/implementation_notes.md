# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/discovered_ray_switchboard.py`.
- Idea-local wrapper: `ideas/registry/i206_discovered_ray_switchboard_network/model.py` calls the registered builder.
- Registry key: `discovered_ray_switchboard_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

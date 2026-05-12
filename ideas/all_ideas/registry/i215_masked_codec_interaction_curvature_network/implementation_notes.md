# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/masked_codec_interaction_curvature.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i215_masked_codec_interaction_curvature_network/model.py` calls the registered builder.
- Registry key: `masked_codec_interaction_curvature_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/phase_specialist_calibration_mixture.py`.
- Idea-local wrapper: `ideas/i212_phase_specialist_calibration_mixture/model.py` calls the registered builder.
- Registry key: `phase_specialist_calibration_mixture`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

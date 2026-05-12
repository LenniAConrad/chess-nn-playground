# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/defender_timing_schedule.py`.
- Idea-local wrapper: `ideas/registry/i205_defender_timing_schedule_network/model.py` calls the registered builder.
- Registry key: `defender_timing_schedule_network`.
- Input contract: simple_18 board tensor only; CRTK metadata remains reporting-only.
- Output contract: one puzzle logit (shape `(B,)`) plus idea-specific diagnostic tensors.

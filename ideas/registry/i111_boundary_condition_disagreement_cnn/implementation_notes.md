# Implementation Notes

- Central code: `src/chess_nn_playground/models/boundary_condition_disagreement_cnn.py`.
- Registry key: `boundary_condition_disagreement_cnn`.
- Idea-local wrapper: `ideas/registry/i111_boundary_condition_disagreement_cnn/model.py`
  (a thin `build_model_from_config` over
  `build_boundary_condition_disagreement_cnn_from_config`; no
  `ResearchPacketProbe` is involved).
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.
- Batch candidate: `Boundary-Condition Disagreement CNN`.
- Board-only model. CRTK / source / engine / verification metadata is
  reporting-only and is not consumed by the model.
- Each conv block holds one set of weights and is dispatched to multiple
  boundary modes by explicit `F.pad(mode=...)` followed by a
  `padding=0` `F.conv2d`. Supported modes: `zeros`, `reflect`,
  `replicate`, `circular`. The default config uses all four; ablations
  that drop a mode are well-defined as long as at least two distinct
  modes remain.
- Normalisation is a `GroupNorm` shared across modes (no running stats),
  so per-stream forward passes do not interact through batch statistics.

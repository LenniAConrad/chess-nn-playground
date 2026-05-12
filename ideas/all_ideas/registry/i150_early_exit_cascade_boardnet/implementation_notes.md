# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/early_exit_cascade_boardnet.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i150_early_exit_cascade_boardnet/model.py`.
- Registry key: `early_exit_cascade_boardnet`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.
- The model is intentionally board-only and does not consume engine,
  verification, source, principal-variation, mate-score, best-move, or
  CRTK metadata as input.
- Cascade hyperparameters available in `config.yaml`:
  - `channels`: trunk width (default 64).
  - `hidden_dim`: exit-head width (default 96).
  - `depth`: residual blocks per stage (default 2).
  - `num_exits`: number of exits in the cascade (default 4). Each adds one
    stage and one exit head.
  - `dropout`: exit-head and residual-block dropout.
  - `use_batchnorm`: toggles `BatchNorm2d` in the trunk.
  - `halt_temperature`: scales the halting logits before `sigmoid`.
  - `prob_floor`: numerical clamp guarding the cascade-weight log-domain
    cumulative product.
- `cascade_multi_exit_loss` exposes the per-exit BCE breakdown for
  ablations; the default trainer runs the standard BCE-with-logits on the
  cascaded `logits`, which already differentiates through every exit and
  every halting gate.

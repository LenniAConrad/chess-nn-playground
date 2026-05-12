# Implementation Notes

- Bespoke implementation:
  `src/chess_nn_playground/models/iterative_logit_refinement_cnn.py`
  (`IterativeLogitRefinementCNN`,
  `build_iterative_logit_refinement_cnn_from_config`).
- Registry key: `iterative_logit_refinement_cnn`
  (`src/chess_nn_playground/models/registry.py`).
- Idea-local wrapper:
  `ideas/registry/i152_iterative_logit_refinement_cnn/model.py` ->
  `build_iterative_logit_refinement_cnn_from_config`.
- Source packet:
  `ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`
  (Candidate 4: Iterative Logit Refinement CNN).
- Trunk depth, refinement step count, and correction hidden width are
  controlled by `depth`, `refinement_steps`, and `correction_hidden`
  in `config.yaml`. The packet's suggested
  `c_t = 0.25 * tanh(raw_c_t)` clamp is exposed as `correction_clamp`
  (default `0.25`).
- Weight-tied correction heads are the default. Set
  `untie_corrections: true` in the model config to instantiate a
  distinct correction head per step (the packet's
  `untied_corrections` ablation).
- This idea is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.

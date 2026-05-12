# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/source_rate_calibrated_objective.py`
  (`SourceRateCalibratedObjectiveNetwork`,
  `build_source_rate_calibrated_objective_from_config`).
- Idea-local wrapper: `ideas/all_ideas/registry/i176_source_rate_calibrated_objective/model.py`
  calls the bespoke builder.
- Registry key: `source_rate_calibrated_objective`
  (`src/chess_nn_playground/models/registry.py`).
- The model emits `tau`, `temp` and `source_rate_soft_indicator` in its
  forward output dict so the rate-calibrated penalty
  (`loss_rate = lambda_fp * relu(near_fp_soft - target_near_fp)^2 +
  lambda_recall * relu(target_recall - puzzle_recall_soft)^2`) can be wired
  on top of the standard `BCEWithLogits` without re-implementing the soft
  rates.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.
- Batch candidate: `Source-Rate Calibrated Objective`.
- Board-only: CRTK / source / engine metadata stays reporting-only and is
  never consumed as model input.

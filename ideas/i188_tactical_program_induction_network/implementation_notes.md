# Implementation Notes

- Central code: `src/chess_nn_playground/models/tactical_program_induction.py`.
- Idea-local wrapper: `ideas/i188_tactical_program_induction_network/model.py`.
- Registry key: `tactical_program_induction_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Tactical Program Induction Network`.
- Bespoke board-only architecture: it consumes only the `simple_18`
  current-board tensor (12 piece planes plus the side-to-move plane
  and rule planes). CRTK / engine / source / verification metadata
  are never used as model input.
- Configurable knobs from `config.yaml`: `channels`, `hidden_dim`,
  `token_dim`, `depth` (board encoder depth), `program_steps`,
  `executor_layers`, `dropout`, `use_batchnorm`, and `ablation` for
  the `bag_of_ops_no_order`, `one_step_program`,
  `no_precondition_scores`, `random_op_labels` switches.

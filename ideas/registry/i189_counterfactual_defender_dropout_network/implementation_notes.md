# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/counterfactual_defender_dropout.py`.
- Idea-local wrapper: `ideas/registry/i189_counterfactual_defender_dropout_network/model.py`.
- Registry key: `counterfactual_defender_dropout_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Counterfactual Defender Dropout Network`.
- Bespoke board-only architecture: it consumes only the
  `simple_18` current-board tensor (12 piece planes plus the
  side-to-move plane and rule planes). CRTK / engine / source /
  verification metadata are never used as model input.
- Configurable knobs from `config.yaml`: `channels`, `hidden_dim`,
  `depth`, `dropout`, `use_batchnorm`, plus the bespoke
  `intervention_dim`, `max_masks`, `topk`, and `ablation` for the
  `random_masks`, `defenders_only`, `no_intervention_head`
  switches.

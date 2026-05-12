# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/critical_square_budget_network.py`
  (`CriticalSquareBudgetNetwork`,
  `build_critical_square_budget_network_from_config`).
- Idea-local wrapper: `ideas/registry/i185_critical_square_budget_network/model.py`
  (`build_model_from_config`).
- Registry key: `critical_square_budget_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.
- Batch candidate: `Critical-Square Budget Network`.
- Input contract: repository `simple_18` board tensor only
  (`(B, 18, 8, 8)`); engine, verification, source, and CRTK metadata
  are never consumed as model input.
- Output contract: dict with `logits` of shape `(B,)` for the
  repository `puzzle_binary` BCE-with-logits trainer; see
  `architecture.md` for the full diagnostic dict.
- Key hyperparameters in `config.yaml`: `channels`, `depth`,
  `hidden_dim`, `dropout`, `budget` (the critical-square budget `K`),
  and `saliency_temperature` (sparsity knob).

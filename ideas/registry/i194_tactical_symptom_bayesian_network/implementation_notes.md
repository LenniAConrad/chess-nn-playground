# Implementation Notes

- Central code: `src/chess_nn_playground/models/tactical_symptom_bayesian_network.py`.
- Idea-local wrapper: `ideas/registry/i194_tactical_symptom_bayesian_network/model.py`.
- Registry key: `tactical_symptom_bayesian_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Tactical Symptom Bayesian Network` (rank 9).
- Bespoke noisy-AND/noisy-OR symptom Bayesian network: per-square 1x1
  symptom readouts pooled by noisy-OR over squares, K symptoms
  combined into J latent causes via noisy-OR with sigmoid weights and
  a per-cause leak, causes aggregated through a learned mixture of
  noisy-OR and noisy-AND, puzzle logit = `logit(puzzle_prob) +
  residual_weight * residual_logit`.
- Ablations: `none`, `linear_symptom_readout`, `no_residual_logit`,
  `symptom_dropout`.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.

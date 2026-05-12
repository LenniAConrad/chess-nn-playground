# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/tropical_constraint_circuit_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i060_tropical_constraint_circuit_network/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_tropical_constraint_circuit_network_from_config`.
- Registry key: `tropical_constraint_circuit_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2046_friday_shanghai_tropical_circuit.md`.
- Input is the simple_18 board tensor only; CRTK/source/engine metadata
  are never consumed by the model. Four fixed coordinate planes
  (rank, file, diag, anti-diag) are concatenated to the board tensor
  before the 1x1 literal-cost convolution.
- Clause weights are stored in low-rank form `a = softplus(U V)` so they
  are nonnegative and the parameter count stays manageable for `L = 2048`
  literals at default config.
- The `ablation` field selects between `none`, `sum_product_clause`,
  `mean_literal_pool`, `literal_square_shuffle`, `high_temperature_softmin`,
  and `material_only_literals`. The `mean_literal_pool` and
  `literal_square_shuffle` ablations use fixed deterministic non-trainable
  buffers (a random projection and a random permutation respectively) so
  the head input dimensionality is unchanged across all ablations.

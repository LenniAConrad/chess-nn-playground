# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/vector_quantized_motif_codebook_net.py`.
- Idea-local wrapper: `ideas/i159_vector_quantized_motif_codebook_net/model.py`.
- Registry key: `vector_quantized_motif_codebook_net`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.
- Batch candidate: `Vector-Quantized Motif Codebook Net`.
- Inputs: simple_18 board tensor only; CRTK / source / engine /
  verification / principal-variation / mate-score / best-move metadata
  is reporting-only and is never consumed by the model.
- Codebook updates use exponential moving averages of cluster size and
  centroid sums (no codebook MSE term is required from the trainer).
  Encoder gradients flow through the straight-through estimator on the
  quantized feature map. Commitment and codebook losses are still
  returned in the output dict so sweeps can opt in to an auxiliary
  objective.
- The default config (`config.yaml`) sets `channels = code_dim = 64`,
  `num_codes = 64`, `hidden_dim = 96`, `depth = 2`, `dropout = 0.1`,
  `commitment_weight = 0.25`, `ema_decay = 0.99`. These can all be
  overridden in the `model:` section of an idea config.

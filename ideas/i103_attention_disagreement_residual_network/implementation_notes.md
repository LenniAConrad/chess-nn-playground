# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/attention_disagreement_residual_network.py`.
- Idea-local wrapper: `ideas/i103_attention_disagreement_residual_network/model.py`.
- Registry key: `attention_disagreement_residual_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.
- Batch candidate: `Attention Disagreement Residual Network`.
- The model is board-only and consumes the `simple_18` tensor exclusively. It
  does not read engine, verification, source, CRTK, or proposal-label metadata.
- Defaults from `config.yaml`: `family_count=4`, `query_count=8`,
  `token_dim=channels=64`, `hidden_dim=96`, `head_hidden=hidden_dim`,
  `dropout=0.1`, single puzzle logit (`num_classes=1`).
- The forward returns `logits` plus diagnostic tensors documented in
  `architecture.md` so trainers can log family-level disagreement signals.

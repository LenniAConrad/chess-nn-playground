# Implementation Notes

- Central code: `src/chess_nn_playground/models/attention_perturbation_sensitivity_network.py`.
- Idea-local wrapper: `ideas/i106_attention_perturbation_sensitivity_network/model.py`.
- Registry key: `attention_perturbation_sensitivity_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.
- Batch candidate: `Attention Perturbation Sensitivity Network`.
- The model is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input.
- One forward pass runs the shared attention encoder five times per sample (one
  base pass plus the four masked variants) so the bottleneck is
  sensitivity-contrast rather than attention as a self-attestation.

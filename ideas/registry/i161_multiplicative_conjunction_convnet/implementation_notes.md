# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/multiplicative_conjunction_convnet.py`.
- Registry key: `multiplicative_conjunction_convnet`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.
- Batch candidate: `Multiplicative Conjunction ConvNet`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The model uses a plain CNN stem followed by paired branch residual blocks where
  `a * b` is an explicit fusion feature alongside `a`, `b`, and `sigmoid(g) * a`.
  Normalization is applied after product fusion.

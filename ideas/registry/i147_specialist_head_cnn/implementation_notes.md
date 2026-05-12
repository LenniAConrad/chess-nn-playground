# Implementation Notes

- Central code: `src/chess_nn_playground/models/specialist_head_cnn.py`.
- Registry key: `specialist_head_cnn`.
- Idea-local wrapper: `ideas/registry/i147_specialist_head_cnn/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.
- Batch candidate: `Specialist-Head CNN`.
- This is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input.

The model follows the packet sketch with a shared residual CNN trunk, fixed
global/center/edge pooling heads, a safe-decoded own/opponent king-zone head, a
material/count head, and a learned fusion MLP over specialist features and
logits.

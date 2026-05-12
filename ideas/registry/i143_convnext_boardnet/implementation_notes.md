# Implementation Notes

- Central code: `src/chess_nn_playground/models/convnext_boardnet.py`.
- Idea-local wrapper: `ideas/registry/i143_convnext_boardnet/model.py` calls
  `build_convnext_boardnet_from_config`.
- Registry key: `convnext_boardnet` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.
- Batch candidate: `ConvNeXt BoardNet`.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
- Config key `hidden_dim` is mapped onto the inverted-MLP hidden
  dimension `mlp_hidden_dim`; the builder enforces
  `mlp_hidden_dim > channels` (auto-bumping if a smaller value is
  supplied, since the inverted MLP must expand).

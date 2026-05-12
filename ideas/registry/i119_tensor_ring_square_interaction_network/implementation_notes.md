# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/tensor_ring_square_interaction_network.py`.
- Idea-local wrapper: `ideas/registry/i119_tensor_ring_square_interaction_network/model.py` (delegates to
  `build_tensor_ring_square_interaction_network_from_config`).
- Registry key: `tensor_ring_square_interaction_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.
- Batch candidate: `Tensor-Ring Square Interaction Network`.
- Input contract: ``simple_18`` board tensor only. CRTK / engine /
  source / verification metadata is reporting-only.
- Default knobs (paper-grade): ``token_dim=64``, ``rank=4``,
  ``orders=(2, 3)``, ``num_patterns=8``, ``cnn_channels=32``,
  ``cnn_depth=2``, ``hidden_dim=96``, ``dropout=0.1``.
- Numerical stability: each cyclic contraction divides ``M_{p, k}`` by
  the number of squares (64) so the trace stays bounded; the head
  receives a ``LayerNorm`` of the concatenated contraction features
  and CNN summary.

# Implementation Notes

- Bespoke model file: `src/chess_nn_playground/models/piece_drop_stability_network.py`
- Bespoke model class: `PieceDropStabilityNetwork`
- Builder: `build_piece_drop_stability_network_from_config`
- Registry key: `piece_drop_stability_network`
- Idea-local wrapper: `ideas/i112_piece_drop_stability_network/model.py`
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`
- Batch candidate: `Piece-Drop Stability Network`
- Board-only: simple_18 input `(B, 18, 8, 8)`. CRTK / source / engine
  metadata is reporting-only and is never consumed as model input.
- The shared `ResearchPacketProbe` scaffold has been removed; the idea
  no longer imports or invokes any research-packet probe code.

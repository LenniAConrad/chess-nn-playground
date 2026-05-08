# Implementation Notes

- Central code: `src/chess_nn_playground/models/source_invariant_puzzle_bottleneck.py`.
- Registry key: `source_invariant_puzzle_bottleneck`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Source-Invariant Puzzle Bottleneck`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input. CRTK/source metadata stays reporting-only.
- The idea-local `model.py` is a thin wrapper around
  `build_source_invariant_puzzle_bottleneck_from_config`, no longer a
  `ResearchPacketProbe` scaffold.

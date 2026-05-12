# Implementation Notes

- Central code: `src/chess_nn_playground/models/global_scratchpad_boardnet.py`.
- Registry key: `global_scratchpad_boardnet`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.
- Batch candidate: `Global Scratchpad BoardNet`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The implementation uses a coordinate-aware CNN stem, learned global memory
  slots initialized from pooled board features, recurrent GRUCell slot updates
  from fixed pooled board summaries, FiLM broadcasts back to square features, and
  a pooled board-plus-memory binary classifier.

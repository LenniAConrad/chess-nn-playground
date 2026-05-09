# Implementation Notes

- Bespoke model file:
  `src/chess_nn_playground/models/prototype_patch_dictionary_network.py`.
- Registry key: `prototype_patch_dictionary_network`.
- Idea-local wrapper:
  `ideas/i117_prototype_patch_dictionary_network/model.py` is a thin
  adapter over
  `build_prototype_patch_dictionary_network_from_config`.
- Source packet:
  `ideas/research_packets/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.
- Batch candidate: `Prototype Patch Dictionary Network`.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
- The dictionary directions `d_k` are exposed parameters; the shared
  rows are used both in the cosine-softmax assignment and in the
  patch reconstruction so the residual and the histogram refer to the
  same set of motif prototypes.

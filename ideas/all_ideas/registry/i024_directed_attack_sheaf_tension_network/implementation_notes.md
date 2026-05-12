# Implementation Notes

- Registry key: `directed_attack_sheaf_tension_network`.
- Source implementation: `src/chess_nn_playground/models/directed_attack_sheaf.py`.
- Idea wrapper: `ideas/all_ideas/registry/i024_directed_attack_sheaf_tension_network/model.py`.
- The wrapper forwards the idea config to `build_directed_attack_sheaf_from_config` and fills the board encoding from `data.encoding`.
- The implementation is board-only and does not consume engine, verification, source, CRTK metadata, or labels as model inputs.
- The model returns a dictionary with `logits` for training plus directed sheaf diagnostics saved by the common prediction artifact path.

The source packet describes a two-class cross-entropy head and maps fine labels `1` and `2` to positive. This idea folder uses the repository puzzle-binary contract for i024 instead: one BCE logit, with fine labels `0` and `1` mapped to non-puzzle and fine label `2` mapped to puzzle.

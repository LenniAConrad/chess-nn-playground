# Implementation Notes

- Registry key: `one_ply_counterfactual_move_landscape_network`.
- Source implementation: `src/chess_nn_playground/models/move_landscape_net.py`.
- Idea wrapper: `ideas/i025_one_ply_counterfactual_move_landscape_network/model.py`.
- The wrapper forwards the idea config to `build_move_landscape_net_from_config` and fills the encoding from `data.encoding`.
- The deterministic enumerator consumes only current-board `simple_18` piece planes, side-to-move plane, castling planes, and en-passant plane.
- The move set is pseudo-legal: it follows piece movement, blockers, captures, promotions, en-passant targets, and optional geometric castling candidates, but does not filter moves by king safety and does not compute mate, stalemate, legal move trees, or engine evaluations.
- The classifier does not receive raw move count unless `use_count_scalar` is explicitly enabled. The default config leaves it disabled; move count is returned only as a diagnostic.

The source packet describes a two-class cross-entropy head and maps fine labels `1` and `2` to positive. This idea folder uses the repository i025 puzzle-binary contract instead: one BCE logit, with fine labels `0` and `1` mapped to non-puzzle and fine label `2` mapped to puzzle.

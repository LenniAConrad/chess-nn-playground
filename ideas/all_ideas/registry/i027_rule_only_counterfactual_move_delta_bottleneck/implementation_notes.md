# Implementation Notes

- Central code: `src/chess_nn_playground/models/counterfactual_delta_bottleneck.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i027_rule_only_counterfactual_move_delta_bottleneck/model.py` (`build_model_from_config`).
- Registry key: `rule_only_counterfactual_move_delta_bottleneck` (registered via `MODEL_BUILDERS` in `src/chess_nn_playground/models/registry.py`; excluded from `RESEARCH_PACKET_MODEL_NAMES`).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0436_tuesday_los_angeles_move_delta_bottleneck.md`.
- Reused primitives: `Simple18BoardAdapter` and `PseudoLegalDeltaEnumerator` from `src/chess_nn_playground/models/move_landscape_net.py` provide the rule-only `simple_18` parser and the deterministic pseudo-legal one-ply move enumerator. Generation is rule-only: pawn pushes/captures (with promotions), knight/king leaper moves, slider rays stopped by the first occupied square, optional pseudo-castling from the rights planes only. No self-check filtering, no checkmate/stalemate/check oracles, no engine evaluation.
- Bespoke modules: `BoardContextEncoder`, slider-path mean (`_ray_path_mean`), `MoveDeltaTupleEncoder`, masked sparsemax (`_masked_sparsemax`), masked entmax-1.5 (`_masked_entmax15`), `MoveConeBottleneck` and `CounterfactualDeltaClassifierHead`.
- Board-only contract: this idea does not consume engine, verification, source, or CRTK metadata as input.
- Memory: dominant tensor is `[B, max_moves, R]`. With the default config (`max_moves=256`, `R=64`) this stays well below the 750k-parameter and 67 MB ceilings recommended by the markdown thesis.

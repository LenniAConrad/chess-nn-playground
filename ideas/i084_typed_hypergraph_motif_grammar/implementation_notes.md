# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/typed_hypergraph_motif_grammar.py`.
  Defines `TypedHypergraphMotifGrammarNet`,
  `CurrentBoardRelationExtractor`, `MotifRelations`, `GrammarConvBlock`,
  and `build_typed_hypergraph_motif_grammar_from_config`.
- Idea-local wrapper: `ideas/i084_typed_hypergraph_motif_grammar/model.py`
  exposes `build_model_from_config(config)` and delegates to the bespoke
  builder.
- Registry key: `typed_hypergraph_motif_grammar`, registered in
  `src/chess_nn_playground/models/registry.py` and removed from
  `RESEARCH_PACKET_MODEL_NAMES`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0757_tuesday_new_york_motif_grammar.md`.
- Inputs: `simple_18` board tensor only; CRTK/source/Stockfish/PV
  metadata is never consumed by the model.
- Configuration knobs: `channels`, `hidden_dim`, `motif_dim`,
  `board_depth`, `grammar_depth`, `max_pieces`, `fusion_mode`
  (`full | board_only | grammar_only | relation_only | terminal_only`),
  `dropout`, `use_batchnorm`.
- Output: a dict with `(B,)` `logits` for the `puzzle_binary` BCE
  trainer plus typed-grammar diagnostics (see `architecture.md` for the
  full list).

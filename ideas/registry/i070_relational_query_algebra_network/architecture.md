# Architecture

`Relational Query Algebra Network` implements the source packet as a differentiable
board-only relational executor. The model treats the current `simple_18` board tensor
as two learned fact tables: a padded occupied-piece table and a dense 64-square table.
It then evaluates learned query blocks over fixed chess-square relations before fusing
those query summaries with a compact CNN board summary.

## Fact Tables

- `PieceTableExtractor` selects up to 32 occupied squares and emits piece facts with
  piece type, side-to-move ownership, color, absolute and side-relative coordinates,
  slider/king/pawn indicators, normalized material value, castling flags, en-passant
  flags, and a board-level material summary.
- `SquareTableBuilder` emits 64 square facts from occupancy, side-to-move ownership,
  coordinates, center/edge/ray features, en-passant/castling context, and a learned
  projection of local board planes.
- A compact CNN trunk provides dense 8x8 convolutional features so the relational
  path is not forced to rediscover local board texture.

## Query Algebra

Each learned query owns piece-left, piece-right, and square predicates plus a static
learned mixture over a fixed relation bank:

- piece-square join: gated piece facts join square facts through mixed square
  relations;
- piece-piece join: two gated piece tables join through mixed relations between
  their occupied squares;
- piece-square-piece semijoin: two piece predicates gather square evidence lying
  between aligned piece pairs.

The join outputs are summarized with signed mean, magnitude mean, max, top-k mean,
log-sum-exp, and support entropy. Query summaries, material features, and CNN summary
features feed a small MLP classifier that returns one puzzle-binary logit plus
diagnostics including relation entropy, query support entropy, join strengths, CNN
energy, material balance, and piece count.

## Ablation Hooks

The implementation exposes the packet's intended mechanism tests through config:
`no_joins`, `relation_shuffle`, `piece_pair_only`, `no_semijoin`,
`static_relation_mix_only`, `mlp_same_params`, and `fact_table_permutation`.

## Implementation Binding

- Registered model name: `relational_query_algebra_network`.
- Source implementation file: `src/chess_nn_playground/models/relational_query_algebra.py`.
- Idea-local wrapper: `ideas/registry/i070_relational_query_algebra_network/model.py`.

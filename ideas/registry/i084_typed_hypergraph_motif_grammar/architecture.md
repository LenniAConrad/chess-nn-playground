# Architecture

`Typed Hypergraph Motif Grammar` (THMG) is a board-only `puzzle_binary`
classifier whose decisive non-linearity is **chart-parsed motif grammar
over typed piece/relation hypergraphs**. It follows the markdown thesis
from
`ideas/research/packets/classic/chess_nn_research_2026-04-28_0757_tuesday_new_york_motif_grammar.md`.

## Mechanism

1. **Compact convolutional board trunk.** A `Conv2d -> BatchNorm -> SiLU`
   stem maps the `simple_18` board tensor `(B, input_channels, 8, 8)` to
   a `(B, channels, 8, 8)` feature field. `board_depth` `GrammarConvBlock`
   residual blocks (two `Conv2d(3x3, padding=1)` with a `BatchNorm` and
   skip connection) refine the trunk without ever down-sampling square
   identity. The pooled summary `pooled = features.mean(dim=(2, 3))`
   feeds the readout.
2. **Typed-relation hypergraph extraction.** A deterministic
   `CurrentBoardRelationExtractor` reads the board planes and produces
   typed nodes and relations:
   - per-piece attribute vectors (color, type one-hot, side-to-move
     polarity, value, square coordinates, slider/jumper/king flags),
   - binary `attacks_piece`, `defends_piece`, `same_color`, `opp_color`,
     `slider_aligned` matrices,
   - per-piece `attacks_square` masks over the 64 squares,
   - per-king `king_zone` masks and per-piece `near_king_piece` flags,
   - ternary `pinned_to_king` and `only_blocker_between` relations,
   - per-piece `loose_piece`, `underdefended_piece`, `high_value_target`,
     and `king_piece` flags,
   - the side-to-move scalar, material balance, and a relation-fact
     count summary.
3. **Piece encoder and pair scorer.** A `LayerNorm -> Linear -> SiLU ->
   Linear` piece encoder embeds each piece-attribute vector into a
   `motif_dim`-dimensional latent. The pair scorer takes
   `(left, right, left*right, |left-right|, geom)` for each ordered
   piece pair, where `geom` is `(same_color, opp_color, attacks_piece,
   defends_piece, |row_delta| + |file_delta|)`. The output is a single
   pair score per `(source, target)` slot, used as the local building
   block of every typed production.
4. **Chart-parsed motif grammar.** The model maintains 11 typed
   productions composed by masked `logsumexp` / `logaddexp` over the
   relation hypergraph, with a learned `production_bias` per rule:

   - `pressure`: opposite-color attacks (`attacks_piece & opp_color`).
   - `loose_target`: pressure on `loose_piece` or `underdefended_piece`
     targets, combined via `logaddexp`.
   - `king_zone_pressure`: pieces whose `attacks_square` reaches a
     square in the opposing king's `king_zone`, restricted to
     `near_king_piece` and aggregated by king via `logsumexp` over
     squares.
   - `pin_shape`: ternary `(pinner, pinned, king)` chains using the
     deterministic `pinned_to_king` relation.
   - `line_pressure`: aligned slider, lone blocker, end pieces composed
     via `only_blocker_between` and `slider_aligned`.
   - `fork_shape`: distinct pressure pairs against `high_value_target`
     pieces, aggregated by attacker.
   - `battery_shape`: aligned same-color stack reinforcing a
     `line_pressure` target.
   - `compromised_defender`: a defender that is itself pinned via
     `pin_shape`, aggregated by king axis.
   - `overload_shape`: two distinct defenders both forced onto loose
     targets via `defends_piece`.
   - `tactical_convergence`: `logaddexp` of `loose_target` co-occurring
     with a `compromised_defender_by_king` and `pressure`
     co-occurring with `overload_by_target` and `king_zone_by_king`.
   - `puzzle_like_motif`: `logaddexp` of tactical convergence near the
     opposing king and forks targeting the king with a `high_value_target`.

   `grammar_depth` controls the chart's compositional depth: at depth
   `<3` the highest-level convergence and puzzle-motif productions are
   suppressed, at depth `<2` second-level productions (`fork_shape`,
   `battery_shape`, `compromised_defender`, `overload_shape`) are
   suppressed, and at depth `<1` the leaf productions are suppressed,
   exposing a depth ablation switch.
5. **Chart statistics and grammar summary.** Each production yields a
   `(score, mask)` pair and a deterministic 5-feature statistic per
   chart: `max / 16`, `logsumexp / 16`, `mean / 16`, `log1p(active_count)
   / 4`, and the masked `sigmoid(score)` density. These are concatenated
   with a 4-feature relation summary (piece count, relation-fact count,
   material balance, side-to-move) and passed through a `LayerNorm` to
   form the `motif_summary`.
6. **Readout.** The puzzle logit comes from
   `LayerNorm -> Linear -> SiLU -> Dropout -> Linear -> SiLU -> Linear`
   over `[pooled_board, motif_summary]`. A separate grammar-only readout
   consumes `motif_summary` alone so the grammar-only ablation from the
   markdown is always available as `grammar_only_logits`. The main
   `logits` tensor has shape `(B,)` for the `puzzle_binary`
   BCE-with-logits trainer. The `fusion_mode` argument
   (`full | board_only | grammar_only | relation_only | terminal_only`)
   gates which streams reach the readout for ablation studies.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the repository
`puzzle_binary` BCE-with-logits trainer. Diagnostic tensors saved to
prediction artefacts include:

- `grammar_only_logits`: `(B,)` ablation logit using only the
  motif/grammar summary (`motif_summary`).
- `motif_summary`: `(B, summary_dim)` `LayerNorm`-normalized chart and
  relation summary that feeds the readout and the grammar-only ablation.
- Per-production strength scalars, each `(B,)`, computed as
  `sigmoid(logsumexp(score) / 8)` over the production's masked
  hyperedges:
  `pressure_motif_strength`, `loose_target_strength`,
  `king_zone_pressure_strength`, `pin_shape_strength`,
  `line_pressure_strength`, `fork_shape_strength`,
  `battery_shape_strength`, `compromised_defender_strength`,
  `overload_shape_strength`, `tactical_convergence_strength`,
  `puzzle_like_motif_strength`.
- `grammar_chart_energy`: `(B,)` mean squared chart strength across all
  11 productions.
- `motif_entropy`: `(B,)` Shannon entropy of the production-mass softmax
  used as a calibration diagnostic.
- `relation_fact_count`, `piece_count`: `(B,)` deterministic counts of
  the typed-relation evidence consumed by the chart.
- `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`, `grammar_composition_depth`: legacy
  packet-family diagnostic aliases produced from the same grammar
  tensors so reporting parity with other packet folders is preserved.

When `return_aux=True` the dict also exposes `production_mass`, the raw
`(B, 11)` `logsumexp` chart masses prior to the softmax used for
`motif_entropy`.

## Leakage Guards

The forward pass consumes only the `simple_18` board tensor. The
packet's forbidden inputs (Stockfish scores, principal variations, node
counts, mate scores, best moves, verification metadata, source labels,
source identity) are never passed to the model. CRTK metadata is
reporting-only.

## Implementation Binding

- Registered model name: `typed_hypergraph_motif_grammar`.
- Source implementation file: `src/chess_nn_playground/models/typed_hypergraph_motif_grammar.py`.
- Idea-local wrapper: `ideas/registry/i084_typed_hypergraph_motif_grammar/model.py`.

# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/move_graph_router.py`.
- Idea-local wrapper: `ideas/registry/a011_bt4_move_graph_router_mixer/model.py`.
- Registered model alias: `bt4_move_graph_router_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=move_graph_router`).
- Source primitive idea: `ideas/registry/p006_move_graph_router`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `move_graph_router`. The tower then
constructs `N` `BT4MixerBlock`s, each of which builds the named mixer
through the `bt4_mixers.build_mixer` factory. The mixer is required by
the BT4 block to be shape-preserving:
`mixer(x).shape == x.shape == (B, C, 8, 8)`; the block raises
`ValueError` otherwise. SqueezeExcite + residual + ReLU wrap the mixer
output without changing its rank.

The MGR mixer flattens the `(B, C, 8, 8)` board to per-square tokens
`(B, 64, C)`, builds a content-derived sparse 0/1 adjacency
`(B, 64, 64)` from two small linear scorers
(`src_score`, `dst_score`) under `@torch.no_grad()` so the mask is
stop-grad, gathers per-edge messages with a shared two-layer GELU
`edge_mlp` over the concat `[x_i, x_j]`, applies the mask, sums per
source, divides by the per-source degree (floored at 1), normalises
with a `LayerNorm`, and reshapes back to `(B, C, 8, 8)`.

## Honest deviation from the source primitive

The source primitive (`p006`) derives its adjacency from the
`simple_18` piece planes, occupancy, side-to-move, and the
precomputed geometric attack tables. A BT4 mixer only sees an
opaque-channel `(B, C, 8, 8)` tensor after the stem and several
residual blocks, so the rule-derived edge set is not reconstructible
here. The mixer therefore substitutes a content-derived thresholded
adjacency for the rule-derived `E_b`. The gather-scatter operator,
the concat-MLP edge function, the stop-gradient discrete mask, and
the degree-normalised aggregation are preserved exactly. This is the
deliberate adaptation called out in the mixer module's docstring; the
A1 / A2 ablations against the `conv` and `attention` baselines (see
`ablations.md`) tell us whether the resulting operator still buys
anything over generic mixers.

## Input contract

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Castling planes, en-passant plane, and side-to-move plane are
all part of the standard `simple_18` encoding; no CRTK metadata, FEN,
Stockfish PV, or source label is read at any point.

## Output contract

The tower's value head emits a single logit per sample. To stay
compatible with the shared puzzle_binary `bce_with_logits` trainer,
the model returns either a `(B,)` tensor or a dict with key `logits`
of shape `(B,)`. The forward smoke test in
`tests/test_idea_registry.py::test_fully_implemented_idea_is_smoke_testable`
runs at batch size 2 against this contract.

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`.
`audit_implementation_kinds.py` detects this as `bespoke_model`,
which matches the `idea.yaml implementation_kind: bespoke_model`
declaration. The tower itself is bespoke code shared across all
`a###_bt4_*_mixer` ideas; each idea pins one specific mixer name as
a controlled-study variable.

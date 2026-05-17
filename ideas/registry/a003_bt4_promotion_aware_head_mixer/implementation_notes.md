# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/promotion_aware_head.py`.
- Idea-local wrapper: `ideas/registry/a003_bt4_promotion_aware_head_mixer/model.py`.
- Registered model alias: `bt4_promotion_aware_head_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=promotion_aware_head`).
- Source primitive idea: `ideas/registry/i246_promotion_aware_head`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `promotion_aware_head`. The tower then
constructs `N` `BT4MixerBlock`s, each of which builds the named mixer
through the `bt4_mixers.build_mixer` factory. The mixer is required by
the BT4 block to be shape-preserving:
`mixer(x).shape == x.shape == (B, C, 8, 8)`; the block raises
`ValueError` otherwise. SqueezeExcite + residual + ReLU wrap the mixer
output without changing its rank.

## Input contract

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Castling planes, en-passant plane, and side-to-move plane are
all part of the standard `simple_18` encoding; no CRTK metadata, FEN,
Stockfish PV, or source label is read at any point. The PAH mixer does
not look at piece planes directly: it operates on the 64 square-tokens
of the BT4 trunk's intermediate feature map, applies four learned
per-type transforms uniformly to every token, and softmax-weights them
through a per-token cross-attention head.

## Output contract

The tower's value head emits a single logit per sample. To stay
compatible with the shared puzzle_binary `bce_with_logits` trainer,
the model returns either a `(B,)` tensor or a dict with key `logits` of
shape `(B,)`. The forward smoke test in
`tests/test_idea_registry.py::test_fully_implemented_idea_is_smoke_testable`
runs at batch size 2 against this contract.

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder directly
and does not delegate to `build_research_packet_probe_from_config`.
`audit_implementation_kinds.py` detects this as `bespoke_model`, which
matches the `idea.yaml implementation_kind: bespoke_model` declaration.
The tower itself is bespoke code shared across all `a###_bt4_*_mixer`
ideas; each idea pins one specific mixer name as a controlled-study
variable.

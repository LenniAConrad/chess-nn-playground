# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/sparse_delta_accumulator.py`.
- Idea-local wrapper: `ideas/registry/a018_bt4_sparse_delta_accumulator_mixer/model.py`.
- Registered model alias: `bt4_sparse_delta_accumulator_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=sparse_delta_accumulator`).
- Source primitive idea: `ideas/registry/p013_sparse_delta_accumulator`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `sparse_delta_accumulator`. The tower then
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
Stockfish PV, or source label is read at any point.

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

## Adapter caveat (SDA-specific)

SDA's defining contract is the *stateful O(|delta|) make/unmake
autograd path*, which is an inference-time property with no static-
batch analogue. The mixer adaptation in
`src/chess_nn_playground/models/architecture/bt4_mixers/sparse_delta_accumulator.py`
reproduces only the analytical fixed point ``h = sum_i W[i]`` followed
by the ClippedReLU and broadcast-fuse step; the make/unmake
statefulness is not expressible inside the `(B, C, 8, 8) -> (B, C, 8, 8)`
mixer contract and is not attempted. Read any null result here as a
falsifier for the static-fixed-point variant of SDA only, not for the
delta-stream variant.

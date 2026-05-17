# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/blocker_reset_ray_scan.py`.
- Idea-local wrapper: `ideas/registry/a025_bt4_blocker_reset_ray_scan_mixer/model.py`.
- Registered model alias: `bt4_blocker_reset_ray_scan_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=blocker_reset_ray_scan`).
- Source primitive idea: `ideas/registry/p020_blocker_reset_ray_scan`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `blocker_reset_ray_scan`. The tower then
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

## Spatial-mixer adaptation

The source primitive (p020) reads occupancy directly off the simple_18
piece planes (a hard binary stop-gradient mask). The BT4 spatial-mixer
contract takes an arbitrary `(B, C, 8, 8)` channel tensor with no
occupancy plane, so the mixer derives a soft occupancy indicator
`O_s = sigmoid(w . x_s + b)` from the per-square channel vector. This
preserves the defining property of the source thesis -- the blocker
mask is generated *inside* the operator and never supplied externally
-- but it is a learned indicator rather than a binary piece-plane read.
The math thesis flags this as one of the documented failure modes to
watch for. Ray geometry itself (the 8 queen-style directions, the per-
direction step indices, and the off-board step mask) is a rule-derived
constant buffer registered at construction time.

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder directly
and does not delegate to `build_research_packet_probe_from_config`.
`audit_implementation_kinds.py` detects this as `bespoke_model`, which
matches the `idea.yaml implementation_kind: bespoke_model` declaration.
The tower itself is bespoke code shared across all `a###_bt4_*_mixer`
ideas; each idea pins one specific mixer name as a controlled-study
variable.

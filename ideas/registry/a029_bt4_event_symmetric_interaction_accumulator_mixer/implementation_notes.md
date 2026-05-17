# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/event_symmetric_interaction_accumulator.py`.
- Idea-local wrapper: `ideas/registry/a029_bt4_event_symmetric_interaction_accumulator_mixer/model.py`.
- Registered model alias: `bt4_event_symmetric_interaction_accumulator_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with
  `mixer=event_symmetric_interaction_accumulator`).
- Source primitive idea: `ideas/registry/p024_event_symmetric_interaction_accumulator`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to
`event_symmetric_interaction_accumulator`. The tower then constructs
`N` `BT4MixerBlock`s, each of which builds the named mixer through
the `bt4_mixers.build_mixer` factory. The mixer is required by the
BT4 block to be shape-preserving:
`mixer(x).shape == x.shape == (B, C, 8, 8)`; the block raises
`ValueError` otherwise. SqueezeExcite + residual + ReLU wrap the
mixer output without changing its rank.

## Input contract

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Castling planes, en-passant plane, and side-to-move plane are
all part of the standard `simple_18` encoding; no CRTK metadata,
FEN, Stockfish PV, or source label is read at any point.

## Output contract

The tower's value head emits a single logit per sample. To stay
compatible with the shared puzzle_binary `bce_with_logits` trainer,
the model returns either a `(B,)` tensor or a dict with key `logits`
of shape `(B,)`. The forward smoke test in
`tests/test_idea_registry.py::test_fully_implemented_idea_is_smoke_testable`
runs at batch size 2 against this contract.

## Spatial-mixer adaptation

The source primitive (p024) is an *additive head* over the i193
trunk that pools the elementary-symmetric stack
`[E^{(1)}; ...; E^{(R)}]` through an MLP to a single delta logit,
gated and added to the trunk's base logit. The BT4 spatial-mixer
contract requires a full `(B, C, 8, 8)` channel tensor back, so the
mixer broadcasts the global symmetric states back to every square
and fuses them with each square's own token via a per-square MLP:

```
y_s = MLP([u_s; E^{(1)}; ...; E^{(R)}])
```

The elementary-symmetric recurrence is preserved exactly:

```
E^{(0)} = 1
for s = 0 .. 63:
    for r = R, R-1, ..., 1:
        E^{(r)} <- E^{(r)} + u_s (.) E^{(r-1)}
```

with each `E^{(r)}` divided by the soft active count raised to the
matching power so the head sees scale-invariant features. The
default `order = R = 2`; `R in {1, 2, 3}` is permitted. Honest
compromise: the broadcast-back fusion is added structure not present
in the pooled-readout head; the elementary-symmetric recurrence
itself is faithful.

The source primitive reads occupancy directly off the simple_18
piece planes (a hard binary mask). The BT4 spatial-mixer contract
takes an arbitrary `(B, C, 8, 8)` channel tensor with no occupancy
plane, so the mixer derives a soft occupancy indicator
`O_s = sigmoid(w . x_s + b)` from the per-square channel vector and
multiplies it into `u_s` before the recurrence. The math thesis
flags this as one of the documented failure modes to watch for (the
in-mixer `zero_occupancy`-style ablation).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`.
`audit_implementation_kinds.py` detects this as `bespoke_model`,
which matches the `idea.yaml implementation_kind: bespoke_model`
declaration. The tower itself is bespoke code shared across all
`a###_bt4_*_mixer` ideas; each idea pins one specific mixer name as
a controlled-study variable.

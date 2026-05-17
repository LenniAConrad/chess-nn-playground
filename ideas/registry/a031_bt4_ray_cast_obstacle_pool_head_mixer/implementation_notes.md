# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/ray_cast_obstacle_pool_head.py`.
- Idea-local wrapper: `ideas/registry/a031_bt4_ray_cast_obstacle_pool_head_mixer/model.py`.
- Registered model alias: `bt4_ray_cast_obstacle_pool_head_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=ray_cast_obstacle_pool_head`).
- Source primitive idea: `ideas/registry/p026_ray_cast_obstacle_pool_head`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `ray_cast_obstacle_pool_head`. The tower
then constructs `N` `BT4MixerBlock`s, each of which builds the named
mixer through the `bt4_mixers.build_mixer` factory. The mixer is
required by the BT4 block to be shape-preserving:
`mixer(x).shape == x.shape == (B, C, 8, 8)`; the block raises
`ValueError` otherwise. SqueezeExcite + residual + ReLU wrap the
mixer output without changing its rank.

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

## Spatial-mixer adaptation

The source primitive (p026) is a *pooling head* over the i193 trunk
that reads the rule-exact occupancy mask from the `simple_18` piece
planes, mean-pools the 8-direction per-square ray stack down to a
single `(NUM_DIRECTIONS * feature_dim)` flat vector, and emits one
gated delta logit added to the trunk's base logit. The BT4 spatial-
mixer contract requires a full `(B, C, 8, 8)` channel tensor back, so:

- The rule-exact piece-plane occupancy is replaced by a soft
  per-square occupancy proxy `O = sigmoid(Conv1x1(X))` learned from
  the channel features themselves. The geometric-decay + running-
  unblocked-product structure is preserved exactly; only the
  source of the occupancy signal changes.
- Instead of mean-pooling the 8-direction stack to a flat vector,
  we keep the per-direction accumulators at full spatial resolution
  `(B, NUM_DIRECTIONS * C, 8, 8)` and project back to `C` channels
  with a `Conv1x1` (`out_proj`). This satisfies the shape-preserving
  mixer contract.
- Per-direction shifts use a zero-padded translate
  (`_shift_along_direction`) so contributions from off-board cells
  are zero by construction; `max_ray_length` is capped at 7 (the
  longest chess ray) so the cumulative product/cumulative sum stay
  bounded.

The geometric-decay accumulator -- the load-bearing idea -- is
faithful: the per-direction prefix sum with running unblocked
weight matches the source primitive's recurrence exactly. Honest
compromise: the soft occupancy proxy is content-based, not rule-
exact, and the per-direction full-resolution stack plus 1x1
projection is added structure not present in the pooled-readout
head. Both compromises are required by the channel-agnostic mixer
contract and are tested by the cross-idea ablations (A1 vs `conv`,
A2 vs `attention`, A3 vs the primitive as a pooled head).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

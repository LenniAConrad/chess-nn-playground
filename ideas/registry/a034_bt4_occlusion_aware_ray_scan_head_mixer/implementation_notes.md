# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/occlusion_aware_ray_scan_head.py`.
- Idea-local wrapper: `ideas/registry/a034_bt4_occlusion_aware_ray_scan_head_mixer/model.py`.
- Registered model alias: `bt4_occlusion_aware_ray_scan_head_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=occlusion_aware_ray_scan_head`).
- Source primitive idea: `ideas/registry/p029_occlusion_aware_ray_scan_head`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `occlusion_aware_ray_scan_head`. The
tower then constructs `N` `BT4MixerBlock`s, each of which builds the
named mixer through the `bt4_mixers.build_mixer` factory. The mixer
is required by the BT4 block to be shape-preserving:
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

The source primitive (p029, OARS) is a *pooling head* over the i193
trunk that runs a selective associative scan along each of the 8
chess directions, projects the per-direction state to a scalar, and
mean-pools to a single delta logit added to the trunk's base logit.
The BT4 spatial-mixer contract requires a full `(B, C, 8, 8)` channel
tensor back, so:

- The per-(square, direction) blocker gate
  `g = sigma(Conv2d(C -> 8)(x))` is preserved exactly. The gate is
  computed once from the raw per-square channel features (fixed-
  feature simplification, identical to the source primitive's
  stability-on-8x8 choice).
- The per-direction iterated scan
  `state_{t+1} = features + g * shift_d(state_t)` is preserved
  exactly with `max_ray_length = 7` (the longest possible chess ray
  on an 8x8 board). The state is initialised at zero; after
  `max_ray_length` steps the state has accumulated the full
  per-direction prefix-sum weighted by the per-step gate product.
- The per-direction read-out `C_d` is preserved as a per-direction
  `Conv2d(C -> C, 1x1)` (instead of the source primitive's
  `Conv2d(C -> 1, 1x1)` followed by mean-pool). The eight
  per-direction outputs are summed into the mixer output
  `y_i = sum_d (C_d state_{i, d})`. This satisfies the
  shape-preserving mixer contract.
- The pooled trunk-fusion path (gate + delta MLPs that consume the
  pooled OARS output together with the trunk diagnostics) is
  dropped, since the BT4 block has no trunk to import; the per-
  square output replaces the pooled scalar.

The selective associative scan with content-dependent blocker
gating -- the load-bearing idea -- is faithful: the gate is a
learned per-(square, direction) sigmoid head on the per-square
channel features; the scan is the same iterated
`state = features + g * shifted_state` over `max_ray_length` steps;
the per-direction read-out is preserved up to the channel-count
swap. Honest compromise: the final mean-pool to a scalar is
replaced by per-direction `Conv2d(C -> C, 1x1)` projections + sum
across directions (required by the channel-agnostic mixer
contract), and the gate is recomputed from raw `x` per block rather
than from the running state (matches the source primitive's
fixed-feature simplification). Both compromises are tested by the
cross-idea ablations (A1 vs `conv`, A2 vs `attention`, A3 vs the
primitive as a pooled head; in-mixer A5 `disable_blocker_gate` and
A8 `shuffle_directions` mirror the primitive's own falsifiers).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

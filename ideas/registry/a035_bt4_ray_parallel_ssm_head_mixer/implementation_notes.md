# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/ray_parallel_ssm_head.py`.
- Idea-local wrapper: `ideas/registry/a035_bt4_ray_parallel_ssm_head_mixer/model.py`.
- Registered model alias: `bt4_ray_parallel_ssm_head_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=ray_parallel_ssm_head`).
- Source primitive idea: `ideas/registry/p030_ray_parallel_ssm_head`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `ray_parallel_ssm_head`. The tower then
constructs `N` `BT4MixerBlock`s, each of which builds the named
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

The source primitive (p030, Ray-SSM) is a *pooling head* over the
i193 trunk that runs a selective state-space scan along each of the
8 chess directions, mean-pools the per-direction state to a feature
vector, and fuses with trunk diagnostics via gate / delta MLPs to a
single delta logit added to the trunk's base logit. The BT4 spatial-
mixer contract requires a full `(B, C, 8, 8)` channel tensor back, so:

- The per-(square, direction, channel) selective scalars `A` and
  `B` are preserved exactly. `A = sigma(A_proj(x))` and
  `B = sigma(B_proj(x))` are 1x1 convs `Conv2d(C -> NUM_DIRECTIONS * C)`
  whose outputs are reshaped to `(B, NUM_DIRECTIONS, C, 8, 8)`.
- The per-direction iterated scan
  `h_{t+1} = A_d * shift_d(h_t) + B_d * x` is preserved exactly with
  `max_ray_length = 7` (the longest possible chess ray on an 8x8
  board). The state is initialised at zero; after `max_ray_length`
  steps the state has accumulated the full per-direction selective
  state-space convolution.
- The per-direction read-out `C[d]` is preserved as a learned
  `(NUM_DIRECTIONS, C)` parameter tensor. The eight per-direction
  outputs are summed into `y_total = sum_d state_d * C[d]` (with
  `C[d]` broadcast across the 8x8 spatial grid). A final
  `Conv2d(C -> C, 1x1)` output projection then maps `y_total` to
  the mixer output. This satisfies the shape-preserving mixer
  contract without dropping the per-direction `C` read-out.
- The pooled trunk-fusion path (mean-pool + LayerNorm + gate /
  delta MLPs over `(feature_dim + 4)` consuming the four trunk
  diagnostics) is dropped, since the BT4 block has no trunk to
  import; the per-square output replaces the pooled scalar.

The selective state-space scan with per-(square, direction, channel)
A and B -- the load-bearing idea -- is faithful: A and B are learned
sigmoid 1x1 convs over the per-square features; the scan is the same
iterated `h = A * shifted_h + B * x` over `max_ray_length` steps; the
per-direction read-out is preserved as a learned `C[d]` per-channel
vector. Honest compromise: the final mean-pool to a feature vector
is dropped (the BT4 block needs a `(B, C, 8, 8)` channel tensor, not
a pooled scalar), and the per-direction `C` is per-direction-only
(matches the source primitive's simplification documented in
`p030/implementation_notes.md`; the spec's full form requires
`C_{i, d}` indexed by both square and direction). Both compromises
are tested by the cross-idea ablations (A1 vs `conv`, A2 vs
`attention`, A3 vs the primitive as a pooled head; in-mixer A5
`disable_selective_A` and A6 `disable_selective_B` mirror the
primitive's own falsifiers).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

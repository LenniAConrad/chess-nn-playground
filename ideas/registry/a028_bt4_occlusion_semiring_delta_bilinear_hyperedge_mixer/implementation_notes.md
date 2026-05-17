# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/occlusion_semiring_delta_bilinear_hyperedge.py`.
- Idea-local wrapper: `ideas/registry/a028_bt4_occlusion_semiring_delta_bilinear_hyperedge_mixer/model.py`.
- Registered model alias: `bt4_occlusion_semiring_delta_bilinear_hyperedge_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with
  `mixer=occlusion_semiring_delta_bilinear_hyperedge`).
- Source primitive idea: `ideas/registry/p023_occlusion_semiring_delta_bilinear_hyperedge`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to
`occlusion_semiring_delta_bilinear_hyperedge`. The tower then
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

The source primitive (p023) is an *additive head* over the i193
trunk that mean-pools the per-square bilinear hyperedge embeddings
to a single delta logit, gated and added to the trunk's base logit.
The BT4 spatial-mixer contract requires a full `(B, C, 8, 8)`
channel tensor back, so the mixer keeps the per-square bilinear
hyperedges instead of pooling them, concatenates the 4 pair
embeddings along the channel axis, and projects back to `C`:

```
out = W_O concat_p (W_L h_{left_p,s} (.) W_R h_{right_p,s})
```

The backward occlusion-semiring recurrence is preserved exactly
along each of the 8 queen-ray directions:

```
h_{r,L} = 0
h_{r,t} = (1 - O_{c_{r,t+1}}) * h_{r,t+1} + V x_{c_{r,t+1}}
```

with the off-board mask applied per-step via the precomputed
`ray_step_mask` buffer. The 4 opposite-direction pairs and the
ordered ray cells are tabulated at construction time and registered
as non-persistent buffers (`ray_step_index`, `ray_step_mask`). Honest
compromise: the per-square bilinear concat + projection is added
structure not present in the pooled-readout head; the backward
recurrence and the opposite-direction bilinear hyperedge themselves
are faithful.

The source primitive reads occupancy directly off the simple_18
piece planes (a hard binary mask). The BT4 spatial-mixer contract
takes an arbitrary `(B, C, 8, 8)` channel tensor with no occupancy
plane, so the mixer derives a soft occupancy indicator
`O_s = sigmoid(occ_proj(x_s))` from the per-square channel vector
and uses `(1 - O)` as the transmittance gate. The math thesis flags
this as one of the documented failure modes to watch for (the
in-mixer `zero_occupancy`/`uniform_occupancy`-style ablation).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

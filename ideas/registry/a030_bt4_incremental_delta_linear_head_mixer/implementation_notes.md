# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/incremental_delta_linear_head.py`.
- Idea-local wrapper: `ideas/registry/a030_bt4_incremental_delta_linear_head_mixer/model.py`.
- Registered model alias: `bt4_incremental_delta_linear_head_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=incremental_delta_linear_head`).
- Source primitive idea: `ideas/registry/p025_incremental_delta_linear_head`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `incremental_delta_linear_head`. The tower
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

The source primitive (p025) is an *additive head* over the i193
trunk that reads a `12 x 64` piece-plane indicator and indexes a
per-(piece-type, square) embedding table `E in R^{12 x 64 x d}` to
emit a single delta logit, gated and added to the trunk's base
logit. The BT4 spatial-mixer contract requires a full
`(B, C, 8, 8)` channel tensor back, so:

- The per-(piece-type, square) embedding axis is absorbed into a
  per-square linear map `W_s : R^C -> R^d` of the channel vector
  (i.e. `E` is realised as 64 distinct linear maps applied to the
  current channel feature, not to a one-hot piece indicator). The
  per-square structure is preserved; the per-piece-type
  factorisation is replaced by a soft channel descriptor.
- The global accumulator `S = sum_s W_s x_s + b_s` is normalised
  through `LayerNorm` and then broadcast back to every square,
  where it is fused with that square's own token via a per-square
  MLP:

  ```
  y_s = MLP([x_s; S])
  ```

The linear-additive accumulator -- the load-bearing idea -- is
faithful: `S` remains linear in the per-square contribution, so
the `O(k)` per-move incremental update property of the source
primitive is preserved in the projection step (the per-square
linear maps are exactly what would be re-summed for the changed
squares). Honest compromise: the broadcast-back fusion is added
structure not present in the pooled-readout head, and the per-
piece-type embedding axis cannot exist in a channel-agnostic
mixer.

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

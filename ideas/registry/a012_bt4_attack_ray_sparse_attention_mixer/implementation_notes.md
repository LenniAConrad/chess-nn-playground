# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/attack_ray_sparse_attention.py`.
- Idea-local wrapper: `ideas/registry/a012_bt4_attack_ray_sparse_attention_mixer/model.py`.
- Registered model alias: `bt4_attack_ray_sparse_attention_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=attack_ray_sparse_attention`).
- Source primitive idea: `ideas/registry/p007_attack_ray_sparse_attention`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `attack_ray_sparse_attention`. The tower
then constructs `N` `BT4MixerBlock`s, each of which builds the named
mixer through the `bt4_mixers.build_mixer` factory. The mixer is
required by the BT4 block to be shape-preserving:
`mixer(x).shape == x.shape == (B, C, 8, 8)`; the block raises
`ValueError` otherwise. SqueezeExcite + residual + ReLU wrap the
mixer output without changing its rank.

The ARSA mixer flattens the `(B, C, 8, 8)` board to per-square tokens
`(B, 64, C)`, derives a content-driven soft occupancy with a small
`occ_score` linear, discretises it to `bool` and detaches it under
`@torch.no_grad()`, then runs a ray-cast first-blocker scan along the
8 sliding-piece directions over precomputed `(64, 8, 7)` ray-step
tables. The resulting `(B, 64, 9)` key index tensor (8 ray neighbours
plus self-edge) is non-differentiable. `q_proj`, `k_proj`, `v_proj`
project the tokens, the per-slot softmax adds a learned per-direction
slot bias, slots with no blocker on the ray are masked to `-inf`, and
the softmax-weighted sum over `v` is normalised by `LayerNorm` before
the tensor is reshaped back to `(B, C, 8, 8)`.

## Honest deviation from the source primitive

The source primitive (`p007`) derives its occupancy from the
`simple_18` piece planes at the network's input. A BT4 mixer only
sees an opaque-channel `(B, C, 8, 8)` tensor after the stem and
several residual blocks, so that occupancy plane is not
reconstructible here. The mixer therefore substitutes a
content-derived thresholded scalar occupancy (under
`@torch.no_grad()`) for the rule-derived one. The ray-cast
first-blocker key-set construction, the 9-slot sparse attention with
masked empty slots, the per-direction slot bias, the stop-grad on the
index tensor, and the masked softmax over a fixed-cardinality key set
are preserved exactly. This is the deliberate adaptation called out
in the mixer module's docstring; the A1 / A2 ablations against the
`conv` and `attention` baselines (see `ablations.md`) tell us whether
the resulting operator still buys anything over generic mixers.

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

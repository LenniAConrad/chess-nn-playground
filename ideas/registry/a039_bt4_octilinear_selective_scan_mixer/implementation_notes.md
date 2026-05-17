# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/octilinear_selective_scan.py`.
- Idea-local wrapper: `ideas/registry/a039_bt4_octilinear_selective_scan_mixer/model.py`.
- Registered model alias: `bt4_octilinear_selective_scan_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=octilinear_selective_scan`).
- Source primitive idea: `ideas/registry/p034_octilinear_selective_scan`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `octilinear_selective_scan`. The tower then
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

The source primitive (p034, OSS) is a *pooling head* over the i193
trunk that ran a Mamba-style selective state-space scan along each
of the 8 chess ray directions, gathered per-direction final-state-
per-square outputs back to `(B, 64, 8 * d)`, fused through
`LayerNorm + Linear + GELU` to a `head_hidden_dim` feature, mean-
pooled (own-piece-weighted + global), and projected to a scalar
gated delta logit fused with the i193 base logit. The BT4 spatial-
mixer contract requires a full `(B, C, 8, 8)` channel tensor back,
so:

- The per-direction selective parameters `A_k(x)` and `B_k(x)` are
  preserved exactly as `nn.Linear(C -> C)` channelwise maps from the
  per-square feature, one pair per direction (8 directions x 2
  projections); `A_k` is wrapped in a `sigmoid` so the multiplicative
  transition stays in `(0, 1)` per channel.
- The per-direction scan
  `h_t = sigmoid(A_k(x_t)) * h_{t-1} + B_k(x_t) * x_t` is preserved
  exactly along the eight rule-aware scan-path tables, computed at
  construction time and held as the non-persistent `scan_paths`
  buffer of shape `(8, num_paths, 8)` with `num_paths = 15` (the
  diagonal maximum); shorter paths are right-padded with `-1` and the
  `valid = path[path >= 0]` mask is applied per-direction inside
  `_scan_direction` so the off-path positions are skipped.
- The 8 per-direction per-square outputs are concatenated to
  `(B, 64, 8 * C)` and fused through
  `LayerNorm(8*C) -> Linear(8*C -> C) -> GELU` before being reshaped
  back to `(B, C, 8, 8)` -- the same fusion shape as the source
  primitive's `LayerNorm + Linear + GELU` to `head_hidden_dim`, but
  with `head_hidden_dim = C` to preserve the spatial-mixer channel
  count.
- The pooled trunk-fusion path (own-piece-weighted mean + global
  mean + scalar gated delta logit MLP) is dropped, since the BT4
  block has no trunk to import; the per-square fused feature
  replaces the pooled scalar.

The Mamba-style selective state-space scan along the 8 chess ray
directions with per-(square, direction, channel) `A_k` and `B_k` --
the load-bearing idea -- is faithful: `A_k` and `B_k` are learned
`Linear(C -> C)` maps over the per-square features, `sigmoid` wraps
`A_k` to keep the transition in `(0, 1)`, the scan is the same
iterated `h = sigmoid(A_k(x)) * h + B_k(x) * x` along the eight
scan-path orderings, and the 8 * C -> C fuser preserves channels.
Honest compromise: the source primitive read piece occupancy off a
`Linear(13) -> d` projection of the simple_18 piece planes, so the
selectivity gate had direct access to "which square holds a piece";
the BT4 spatial-mixer contract takes an arbitrary `(B, C, 8, 8)`
channel tensor, so the gate must rediscover occupancy from whatever
the trunk has encoded into the channel features. Both compromises
are tested by the cross-idea ablations (A1 vs `conv`, A2 vs
`attention`, A3 vs the primitive as a pooled head; in-mixer A5
`fixed_transition` and A6 `single_direction` mirror the primitive's
own falsifiers).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

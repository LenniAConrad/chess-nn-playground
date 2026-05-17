# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/incremental_latent_accumulator_head.py`.
- Idea-local wrapper: `ideas/registry/a033_bt4_incremental_latent_accumulator_head_mixer/model.py`.
- Registered model alias: `bt4_incremental_latent_accumulator_head_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=incremental_latent_accumulator_head`).
- Source primitive idea: `ideas/registry/p028_incremental_latent_accumulator_head`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `incremental_latent_accumulator_head`. The
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

The source primitive (p028) is a *pooling head* over the i193 trunk
that reads the simple_18 `(B, 12, 64)` piece-plane indicator and the
own-king square `k in {0, ..., 64}` directly, builds two embedding-
table accumulators

```
h_global = sum_{(t, s) : x_{t, s} = 1} G_{t, s}
h_king   = sum_{(t, s) : x_{t, s} = 1} K_{k, t, s}
```

with `G in R^{12 x 64 x global_dim}` and
`K in R^{65 x 12 x 64 x king_dim}`, applies a `LayerNorm -> Linear ->
GELU -> Linear` MLP `phi` to `[h_global, h_king]`, and emits a
single gated delta logit added to the trunk's base logit. The BT4
spatial-mixer contract requires a full `(B, C, 8, 8)` channel tensor
back, so:

- The per-(piece-type, square) embedding table
  `G in R^{12 x 64 x global_dim}` is replaced by a
  `Linear(C -> latent_dim)` projection of the per-square channel
  features plus a learned per-square bias
  `global_square in R^{64 x latent_dim}`. The
  permutation-structured accumulation `h_global = sum_s g_s` is
  preserved exactly; only the per-piece-type readout is replaced by
  a per-square learned mixing of the channel features.
- The rule-exact own-king-square anchor `k in {0, ..., 64}` is
  replaced by a *learned* soft-argmax saliency square computed from
  a `Conv2d(C -> 1)` over the channel features, normalised by a
  64-way softmax over the squares. The anchor-conditioned embedding
  row is then a soft mix
  `anchor_embed = einsum(anchor_w, king_anchor_table)` of the
  `(64, latent_dim)` learned `king_anchor_table`. The
  accumulate-then-anchor structure is preserved; only the rule-exact
  king-square readout is replaced by a soft-argmax over learned
  channel features.
- Instead of collapsing the whole board to a single latent and
  emitting one logit, the mixer keeps `h_global` and `h_king` as
  `(B, latent_dim)` board-level latents, broadcasts them back to
  every square as `(B, 64, latent_dim)`, concatenates them with the
  per-square channel features `tokens in R^{B x 64 x C}`, and feeds
  the result through the `phi` MLP `LayerNorm -> Linear -> GELU ->
  Linear` to project back to `C` channels per square. This satisfies
  the shape-preserving mixer contract.
- The trunk-diagnostic fusion (concatenate four scalar diagnostics
  with the pooled latent before the gate / delta MLPs) is dropped,
  since the BT4 block has no trunk to import. The gate / delta MLP
  path is replaced by the per-square `phi` lift followed by the BT4
  block's SqueezeExcite + residual + ReLU.
- The `phi` MLP is preserved with the same `LayerNorm -> Linear ->
  GELU -> Linear` shape as the source primitive, with the input
  dimension widened to `2 * latent_dim + C` (broadcast latents
  concatenated with per-square channel features) and the output
  dimension set to `C` (the channel width) instead of a single
  scalar.

The permutation-structured pooled accumulation followed by a
context-anchored re-accumulation and a non-linear `phi` lift -- the
load-bearing idea -- is faithful: `h_global` is a true permutation-
invariant sum-pool over the 64 squares; `h_king` re-pools with an
additive anchor row indexed by a soft anchor selection; `phi` is the
same `LayerNorm -> Linear -> GELU -> Linear` shape as the primitive.
Honest compromise: the per-piece-type embedding table is replaced by
a `Linear(C -> latent_dim)` over learned channel features (the mixer
cannot read `simple_18` piece planes), the rule-exact king square is
replaced by a learned soft-argmax saliency square (no piece-plane
readout), and the anchor-conditioned embedding table is `(64,
latent_dim)` instead of `(65, 12, 64, king_dim)` (no piece-type
sub-indexing, no separate "no king" row). All three compromises are
required by the channel-agnostic mixer contract and are tested by
the cross-idea ablations (A1 vs `conv`, A2 vs `attention`, A3 vs the
primitive as a pooled head with the rule-exact king-square +
piece-type indicator).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/sparse_legal_move_router_head.py`.
- Idea-local wrapper: `ideas/registry/a032_bt4_sparse_legal_move_router_head_mixer/model.py`.
- Registered model alias: `bt4_sparse_legal_move_router_head_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=sparse_legal_move_router_head`).
- Source primitive idea: `ideas/registry/p027_sparse_legal_move_router_head`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `sparse_legal_move_router_head`. The tower
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

The source primitive (p027) is a *pooling head* over the i193 trunk
that consumes a per-square embedding stack `(B, 64, square_embed_dim)`
built from a piece-type lookup plus a positional embedding, builds a
rule-exact `(B, 64, 64)` legal-move adjacency from the `simple_18`
piece planes via `compute_legal_move_adjacency`, runs one round of
masked attention with the trunk diagnostics fused in, mean-pools the
routed `(B, 64, attn_dim)` tensor to `(B, attn_dim)`, and emits a
gated delta logit added to the trunk's base logit. The BT4 spatial-
mixer contract requires a full `(B, C, 8, 8)` channel tensor back, so:

- The rule-exact `(B, 64, 64)` legal-move adjacency is replaced by a
  static `(64, 64)` chess-geometry support `S` (union of slider rays
  unobstructed, knight L-jumps, king and pawn step shapes) plus a
  *learned* per-(source, target) gate
  `g = sigmoid(theta), theta in R^{64 x 64}` applied as a log-bias
  inside the masked softmax. Off-support edges receive `-inf`
  logits; on-support edges receive `+ log(g_{i, j})` as an additive
  log-bias so the router can suppress an edge below numerical
  resolution but never attends off-support. The chess-structured
  sparsity prior and the masked-softmax aggregator -- the
  load-bearing ideas -- are preserved; only the rule-exact-per-batch
  vs static-plus-learned-gate distinction changes.
- Instead of mean-pooling the routed `(B, 64, attn_dim)` tensor to a
  flat vector, the mixer keeps the routed `(B, 64, C)` tensor at
  full spatial resolution and projects back to `C` channels with a
  `Linear` (`out_proj`). The token rearrangement is `flatten(2) ->
  transpose(1, 2) -> norm -> +pos`, the masked attention runs over
  the 64 tokens, and the routed tokens are reshaped back to
  `(B, C, 8, 8)`. This satisfies the shape-preserving mixer
  contract.
- The trunk-diagnostic fusion (concatenate four scalar diagnostics
  with the pooled router output before the gate / delta MLPs) is
  dropped, since the BT4 block has no trunk to import. The gate /
  delta MLP path is replaced by the standard `out_proj` linear
  back to `C` channels followed by the BT4 block's
  SqueezeExcite + residual + ReLU.
- The self-loop term `M_{i, i} = 1` is preserved in the static
  support so that every row of the support has at least one
  on-support target and the softmax never NaNs even on empty
  rows.

The masked-softmax aggregator with chess-structured sparsity --
the load-bearing idea -- is faithful: off-support edges receive
`-inf` logits exactly, and the on-support log-gate bias matches
the source primitive's per-edge weighting role. Honest
compromise: the adjacency is no longer rule-exact (slider rays
are unobstructed and the per-batch piece occupancy is replaced by
a learned per-edge gate over a fixed support), and the per-token
full-resolution output plus 1x1 projection is added structure
not present in the pooled-readout head. Both compromises are
required by the channel-agnostic mixer contract and are tested by
the cross-idea ablations (A1 vs `conv`, A2 vs `attention`, A3 vs
the primitive as a pooled head with rule-exact adjacency).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

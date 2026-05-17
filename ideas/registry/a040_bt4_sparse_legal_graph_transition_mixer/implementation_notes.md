# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/sparse_legal_graph_transition.py`.
- Idea-local wrapper: `ideas/registry/a040_bt4_sparse_legal_graph_transition_mixer/model.py`.
- Registered model alias: `bt4_sparse_legal_graph_transition_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=sparse_legal_graph_transition`).
- Source primitive idea: `ideas/registry/p035_sparse_legal_graph_transition`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `sparse_legal_graph_transition`. The tower
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

The source primitive (p035, SLMGT) is a *pooling head* over the i193
trunk that built the binary adjacency `A(x)` directly from the
`simple_18` board with piece-specific blocker resolution (the
blocker-resolved legal-move graph), ran the joint edge function
`phi(X_i, X_j) = LayerNorm(ReLU(W_self X_i + W_neighbor X_j +
W_interact (X_i (.) X_j)))` over that adjacency, mean-aggregated
per source square (`Y[i] = (1 / max(deg(i), 1)) sum_j A[i, j]
phi(X_i, X_j)`), pooled, and projected to a gated scalar delta
logit fused with the i193 base logit. The BT4 spatial-mixer contract
requires a full `(B, C, 8, 8)` channel tensor back, so:

- The joint edge function `phi` is preserved exactly: `W_self,
  W_neighbor, W_interact` are `nn.Linear(C -> d_edge)` (default
  `d_edge = C`) channelwise maps over the per-square feature, the
  Hadamard interaction `pair = feats.unsqueeze(2) *
  feats.unsqueeze(1)` is materialised as the explicit `(B, 64, 64,
  C)` pair tensor, the per-edge nonlinearity is the same `ReLU` +
  `LayerNorm(d_edge)` body.
- The hard-binary chess-rule mask is replaced with the *static*
  union of knight, king, and sliding-piece reach (zero diagonal),
  computed at construction time and held as the non-persistent
  `adjacency` buffer. The blocker-resolved per-board legal-move
  adjacency that the source primitive used is unavailable to the
  mixer (it would require `simple_18` access, which the BT4 block
  cannot provide); the static union is the strict superset of the
  blocker-resolved graph. The `inv_degree = 1 / max(degree, 1)`
  buffer is precomputed once.
- The degree-normalised mean aggregation `Y[i] = (1 /
  max(deg(i), 1)) sum_j A[i, j] phi(X_i, X_j)` is preserved
  exactly via the einsum `agg = einsum("bij,bijd->bid", adj, phi)`
  followed by `agg = agg * inv_degree.view(1, -1, 1)`.
- The aggregated `(B, 64, d_edge)` per-square features are
  projected back to `C` via `out_proj = nn.Linear(d_edge -> C)` and
  reshaped to `(B, C, 8, 8)` -- the per-square edge feature
  replaces the source primitive's pooled scalar.
- The pooled trunk-fusion path (per-sample edge-magnitude summary,
  gated scalar delta logit MLP) is dropped, since the BT4 block has
  no trunk to import; the per-square edge feature is returned
  directly.

The joint non-separable edge function with the Hadamard interaction
term, the hard-binary chess-rule mask, the degree-normalised mean
aggregation, and the `LayerNorm` after `ReLU` -- the load-bearing
idea -- are faithful: `W_self`, `W_neighbor`, `W_interact` are
learned `Linear(C -> d_edge)` maps, the Hadamard pair tensor is
materialised explicitly, the aggregation einsum and inverse-degree
weighting are exact. Honest compromise: the source primitive built
`A(x)` from the `simple_18` piece planes with blocker resolution,
so the operator saw the per-board *legal-move* graph; the BT4
spatial-mixer contract takes an arbitrary `(B, C, 8, 8)` channel
tensor with no piece-plane access, so the static union-of-moves
graph stands in for the blocker-resolved legal-move graph. The
static mask is the chess-rule superset of the blocker-resolved
mask, so every edge in the legal-move graph is preserved; what is
lost is the position-conditioned removal of blocked edges. Both
compromises are tested by the cross-idea ablations (A1 vs `conv`,
A2 vs `attention`, A3 vs the primitive as a pooled head; in-mixer
A5 `separable_phi`, A6 `uniform_adjacency`, A7 `shuffle_adjacency`
mirror the primitive's own falsifiers).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

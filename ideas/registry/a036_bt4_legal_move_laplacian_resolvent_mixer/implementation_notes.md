# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/legal_move_laplacian_resolvent.py`.
- Idea-local wrapper: `ideas/registry/a036_bt4_legal_move_laplacian_resolvent_mixer/model.py`.
- Registered model alias: `bt4_legal_move_laplacian_resolvent_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=legal_move_laplacian_resolvent`).
- Source primitive idea: `ideas/registry/p031_legal_move_laplacian_resolvent`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `legal_move_laplacian_resolvent`. The tower
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
the model returns a dict with key `logits` of shape `(B,)`. The
forward smoke test in
`tests/test_idea_registry.py::test_fully_implemented_idea_is_smoke_testable`
runs at batch size 2 against this contract.

## Spatial-mixer adaptation

The source primitive (p031, LM-LPP) is an *i193-additive head* that
builds the pseudo-legal move adjacency `A(x)` analytically from the
`simple_18` piece planes (with sliding-piece blocker resolution and
edge-dropping for own-color targets), forms the signed Laplacian
`L(x) = D(x) - W(x)` weighted by a learned per-piece scalar, applies
the truncated Neumann-series resolvent `Y = sum_k alpha^k L^k X * Theta`,
pools to a feature vector, and fuses with the i193 trunk diagnostics
via gate / delta MLPs to a single delta logit added to the trunk's
base logit. The BT4 spatial-mixer contract requires a full
`(B, C, 8, 8)` channel tensor back, so:

- The Laplacian construction, the Neumann partial sum, the
  `tanh`-bounded `alpha`, and the `Theta` mixing matrix are preserved
  exactly. `alpha = alpha_init * tanh(alpha_logit)` with
  `alpha_init = 0.25` (matches the source primitive's conservative
  default). The Laplacian is rescaled by `max(rowsum, 1)` for
  spectral safety (the source primitive notes power-iteration
  spectral clipping as a future upgrade; the row-degree scaling is
  the conservative substitute used here).
- The static adjacency `A_static in {0, 1}^{64 x 64}` is the
  position-independent union of knight offsets, king offsets, and
  the four sliding-piece alignments (rank, file, both diagonals).
  It is registered as a non-persistent buffer at construction
  time. **Honest compromise** (versus the source primitive's
  occupancy-blocked, piece-typed adjacency): the BT4 mixer only
  receives the `(B, C, 8, 8)` channel feature map, not the discrete
  piece planes, so the occupancy-blocked, piece-typed adjacency
  cannot be computed here.
- Content dependence is restored via a learned per-square scalar
  `w(x) = softplus(MLP(X))` (a small two-layer MLP over the per-
  square feature vector with GELU activation). This scalar plays
  the role of `w(piece(i, x))` in the thesis: it scales each row of
  `A_static` by the learned weight of its own square, giving the
  weighted adjacency `W(x) = diag(w(x)) @ A_static`. The signed
  Laplacian `L = D - W` is then formed exactly as in the thesis.
- The per-square output replaces the pooled scalar: `Y = sum_k
  alpha^k L^k X` is taken per square, projected through `Theta`, and
  reshaped back to `(B, C, 8, 8)` -- no mean-pool to a feature
  vector. The pooled trunk-fusion path (mean-pool + gate / delta
  MLPs over the i193 trunk diagnostics) is dropped, since the BT4
  block has no trunk to import.
- LayerNorm on the per-square feature vector before the per-square
  weight MLP and the Neumann scan is added as the only deviation
  from the source primitive's per-square feature projection (which
  ran a linear projection of the simple_18 piece-existence channels
  + side-to-move plane directly into the `d`-dimensional per-square
  feature). The LayerNorm keeps the per-square feature norms stable
  across blocks; without it the multi-hop Neumann scan can amplify
  per-square magnitudes geometrically across the K terms.

The truncated Neumann partial sum with `tanh`-bounded `alpha` and
the `Theta` mixing matrix -- the load-bearing math -- is faithful:
`alpha` is bounded by `alpha_init = 0.25` via the `tanh` envelope and
the Laplacian is rescaled by `max(rowsum, 1)`, so the partial sum is
bounded for any finite K. Both compromises (no occupancy-based
blocker resolution, no per-piece-type weighting; per-square scalar
instead) are tested by the cross-idea ablations (A1 vs `conv`, A2 vs
`attention`, A3 vs the primitive as an additive head on the i193
trunk; in-mixer A5 `k1_gat_rebrand`, A6 `zero_alpha`, A7
`uniform_piece_weights`, A8 `shuffle_adjacency` mirror the
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

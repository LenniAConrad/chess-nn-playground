# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/dynamic_adjacency_gating.py`.
- Idea-local wrapper: `ideas/registry/a037_bt4_dynamic_adjacency_gating_mixer/model.py`.
- Registered model alias: `bt4_dynamic_adjacency_gating_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=dynamic_adjacency_gating`).
- Source primitive idea: `ideas/registry/p032_dynamic_adjacency_gating`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `dynamic_adjacency_gating`. The tower
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

The source primitive (p032, DAG) is an *i193-additive head* that
builds the position-specific binary adjacency `A(x)` from the
`simple_18` piece planes (with sliding-piece blocker resolution),
intersects it per move type to get
`A_t(x) = A(x) ⊙ 1[move_type(i, j) = t]` for
`t in {RANK, FILE, DIAG, ANTIDIAG, KNIGHT, KING, PAWN_PUSH, PAWN_CAPTURE}`,
applies a per-type linear projection `W_t Z` with a learned per-
square seed feature `Z`, pools the masked aggregation across types,
and fuses with the i193 trunk diagnostics via gate / delta MLPs to a
single delta logit added to the trunk's base logit. The BT4 spatial-
mixer contract requires a full `(B, C, 8, 8)` channel tensor back,
so:

- The hard binary per-type masks, the per-type linear projections
  `W_t`, the final `out_proj` mixing matrix, and the multiplicative
  `g ⊙ Wx` gating form are preserved exactly. The mask is binary
  by design; the gradient of an illegal-edge cell is zero by
  construction.
- The static per-type masks `M_t in {0, 1}^{64 x 64}` are the
  position-independent chess-rule reach geometries: knight, king,
  rank, file, diagonal, antidiagonal. The self-loop is zeroed for
  every type. The masks are registered as a non-persistent buffer
  at construction time. **Honest compromise** (versus the source
  primitive's occupancy-blocked, position-specific adjacency): the
  BT4 mixer only receives the `(B, C, 8, 8)` channel feature map,
  not the discrete piece planes, so the occupancy-blocked, position-
  specific adjacency cannot be computed here. The pawn_push and
  pawn_capture types from the source primitive are dropped (they
  require side-to-move and piece-type discrimination from discrete
  planes); `T = 6` here vs `T = 8` in the source primitive's head
  form.
- Content dependence is restored via a learned per-square per-type
  sigmoid gate `g_t(X_i) = sigmoid(Linear(C, T) X_i)_t in (0, 1)`,
  mirroring the source primitive's "`G(x) ⊙ Wx`" gating form. The
  gate replaces the position-specific binary edge weight in the
  source primitive: where the source primitive zeros illegal edges
  by hard mask intersection, the mixer zeros (or attenuates) entire
  type slots that the per-square content does not endorse.
- The per-square output replaces the pooled scalar: each
  `Y_t = M_t @ (W_t Z)` is computed per square, gated by `g_t(Z_i)`,
  summed across types, projected through `out_proj`, and reshaped
  back to `(B, C, 8, 8)` -- no mean-pool to a feature vector. The
  pooled trunk-fusion path (mean-pool + gate / delta MLPs over the
  i193 trunk diagnostics) is dropped, since the BT4 block has no
  trunk to import.
- LayerNorm on the per-square feature vector before the per-square
  gate and the per-type projections is the only deviation from the
  source primitive's per-square feature projection (which ran a
  linear projection of the simple_18 piece-existence channels +
  side-to-move plane directly into a `feature_dim`-dimensional per-
  square feature). The LayerNorm keeps the per-square feature norms
  stable across blocks; without it the sigmoid gate can saturate
  asymmetrically as the residual stream's magnitude drifts through
  the tower.

The hard binary mask, the per-type projection specialisation, the
multiplicative gating, and the `out_proj` mixing matrix -- the load-
bearing math -- are faithful. Both compromises (no occupancy-based
blocker resolution, no pawn move types) are tested by the cross-idea
ablations (A1 vs `conv`, A2 vs `attention`, A3 vs the primitive as
an additive head on the i193 trunk; in-mixer A5
`single_move_type`, A6 `uniform_gate`, A7 `uniform_adjacency`, A8
`shuffle_adjacency` mirror the primitive's own falsifiers).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

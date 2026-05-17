# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/move_kernel_operator.py`.
- Idea-local wrapper: `ideas/registry/a038_bt4_move_kernel_operator_mixer/model.py`.
- Registered model alias: `bt4_move_kernel_operator_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=move_kernel_operator`).
- Source primitive idea: `ideas/registry/p033_move_kernel_operator`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `move_kernel_operator`. The tower then
constructs `N` `BT4MixerBlock`s, each of which builds the named mixer
through the `bt4_mixers.build_mixer` factory. The mixer is required
by the BT4 block to be shape-preserving:
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

The source primitive (p033, MKO) is defined directly on a per-square
seed feature tensor `X in R^{B x 64 x d}` with the canonical operator
`Y[i] = sum_t sum_{j : M_t[i, j] = 1} (W_t X)[j]` over six static
chess-rule reach types `t in {KNIGHT, RANK, FILE, DIAG, ANTIDIAG, KING}`.
The BT4 spatial-mixer contract requires a full `(B, C, 8, 8)` channel
tensor back, so:

- The hard binary per-type reach masks, the per-type matrix-valued
  linear projections `W_t`, and the final `out_proj` mixing matrix
  are preserved exactly. The mask is binary by design; the gradient
  of an illegal-edge cell is zero by construction.
- The static per-type masks `M_t in {0, 1}^{64 x 64}` are the
  position-independent chess-rule reach geometries: knight, rank,
  file, diagonal, antidiagonal, king. The self-loop is zeroed for
  every type. The masks are registered as a non-persistent buffer
  at construction time. This matches the source primitive's framing
  exactly -- MKO is occlusion-free by design; no blocker resolution
  is required and no simple_18 piece planes are consulted by the
  mixer.
- The per-square seed feature `X` is the BT4 block's `(B, C, 8, 8)`
  channel feature vector flattened to `(B, 64, C)`, rather than a
  `Linear(13)` projection of simple_18 piece planes. This is the
  only deviation from the source primitive's head form; channels `C`
  are preserved end to end and no pool-to-scalar is performed (the
  per-square output is required by the BT4 mixer contract).
- LayerNorm on the per-square feature vector before the per-type
  projections is the only structural addition versus the source
  primitive's per-square feature projection. The LayerNorm keeps
  the per-square feature norms stable across blocks; without it the
  per-type projection outputs can drift asymmetrically as the
  residual stream's magnitude changes through the tower.
- The pooled trunk-fusion path (mean-pool + gate / delta MLPs over
  the i193 trunk diagnostics) is dropped, since the BT4 block has
  no trunk to import.

The hard binary mask, the per-type matrix-valued projection
specialisation, and the `out_proj` mixing matrix -- the load-bearing
math -- are faithful. The only compromise (cosmetic per-square seed
substitution) is tested by the cross-idea ablations (A1 vs `conv`,
A2 vs `attention`, A3 vs the primitive as an additive head on the
i193 trunk; in-mixer A5 `shared_kernel`, A6 `scalar_per_type`, A7
`shuffle_features` mirror the primitive's own falsifiers).

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder
directly and does not delegate to
`build_research_packet_probe_from_config`. `audit_implementation_kinds.py`
detects this as `bespoke_model`, which matches the
`idea.yaml implementation_kind: bespoke_model` declaration. The tower
itself is bespoke code shared across all `a###_bt4_*_mixer` ideas;
each idea pins one specific mixer name as a controlled-study
variable.

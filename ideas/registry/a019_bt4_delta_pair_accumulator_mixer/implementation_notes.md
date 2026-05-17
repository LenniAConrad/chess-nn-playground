# Implementation Notes

- Central tower code: `src/chess_nn_playground/models/architecture/bt4_primitive_mixer.py`
  (`BT4PrimitiveMixerNet`, `build_bt4_primitive_mixer_from_config`).
- Mixer code: `src/chess_nn_playground/models/architecture/bt4_mixers/delta_pair_accumulator.py`.
- Idea-local wrapper: `ideas/registry/a019_bt4_delta_pair_accumulator_mixer/model.py`.
- Registered model alias: `bt4_delta_pair_accumulator_mixer`
  (resolved by `_bt4_alias_to_mixer` in
  `src/chess_nn_playground/models/registry.py` to
  `bt4_primitive_mixer` with `mixer=delta_pair_accumulator`).
- Source primitive idea: `ideas/registry/p014_delta_pair_accumulator`.

## Wiring

`model.py` calls `build_bt4_primitive_mixer_from_config` with
`model.mixer` defaulted to `delta_pair_accumulator`. The tower then
constructs `N` `BT4MixerBlock`s, each of which builds the named mixer
through the `bt4_mixers.build_mixer` factory. The mixer is required by
the BT4 block to be shape-preserving:
`mixer(x).shape == x.shape == (B, C, 8, 8)`; the block raises
`ValueError` otherwise. SqueezeExcite + residual + ReLU wrap the mixer
output without changing its rank.

## Input contract

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Castling planes, en-passant plane, and side-to-move plane are
all part of the standard `simple_18` encoding; no CRTK metadata, FEN,
Stockfish PV, or source label is read at any point.

## Output contract

The tower's value head emits a single logit per sample. To stay
compatible with the shared puzzle_binary `bce_with_logits` trainer,
the model returns either a `(B,)` tensor or a dict with key `logits` of
shape `(B,)`. The forward smoke test in
`tests/test_idea_registry.py::test_fully_implemented_idea_is_smoke_testable`
runs at batch size 2 against this contract.

## Why this is a `bespoke_model`, not a probe variant

The wrapper imports the bespoke `BT4PrimitiveMixerNet` builder directly
and does not delegate to `build_research_packet_probe_from_config`.
`audit_implementation_kinds.py` detects this as `bespoke_model`, which
matches the `idea.yaml implementation_kind: bespoke_model` declaration.
The tower itself is bespoke code shared across all `a###_bt4_*_mixer`
ideas; each idea pins one specific mixer name as a controlled-study
variable.

## Adapter caveat (DPA-specific)

DPA's source-primitive definition is the analytical fixed point
``A(S) = sum_i u_i + sum_{(i,j) in E(S)} W_{type(i),type(j),dsq(i,j)}``
over the active piece-square set, paired with a stateful O(|delta|)
make/unmake autograd path at inference time. The static-batch mixer
adaptation in
`src/chess_nn_playground/models/architecture/bt4_mixers/delta_pair_accumulator.py`
makes two adaptations:

1. The piece-type pair table ``W_{type(i),type(j)}`` is replaced by a
   bilinear of the per-square (src, dst) features projected to
   `pair_dim`, because the BT4 trunk's `C` channels carry no piece-type
   semantics that could index a finite piece-type table.
2. The pair edge set `E(S)` is the rule-derived rank/file/diagonal
   alignment mask over all 64 squares (a fixed buffer, no gradient).
   The source's occupancy-conditioned subset is not recoverable from
   the BT4 trunk's `C` channels and is not attempted.

The (rank_diff, file_diff) `dsq` conditioning, the alignment-restricted
enumeration, the degree-normalised scatter, and the first-order
accumulator broadcast are preserved verbatim from the source. Read any
null result here as a falsifier for this static-fixed-point bilinear
adapter only, not for the make/unmake delta-stream variant of DPA.

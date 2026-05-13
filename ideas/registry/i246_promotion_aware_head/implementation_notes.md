# Implementation Notes — Promotion-Aware Head (i246)

## Module layout

- Model module:
  `src/chess_nn_playground/models/primitives/promotion_aware_head.py`
- Registry key: `promotion_aware_head`
  (`src/chess_nn_playground/models/registry.py`)
- Idea-local entry points: `model.py` and `train.py` in this folder.

## Inputs

- Model input is the simple_18 board tensor `(B, 18, 8, 8)` produced by the
  shared trainer via `fen_to_simple_18`. No additional `batch["..."]` keys
  are required.
- Near-promotion pawn identification and the four promotion substitutions
  are derived analytically from the piece planes (planes 0 and 6 for own
  pawn presence, plane 12 for side-to-move). The substitution is a one-hot
  piece edit — no `python-chess` call is made inside forward, so the
  model has no per-step CPU bottleneck even in the worst case.
- Castling and en-passant planes are not touched by the substitution. The
  underlying chess move would void castling rights for the moved pawn's
  side (irrelevant — pawns can't castle) and would reset en-passant
  (the new feature is not enabled by promotion), but the simple_18 trunk
  is not state-machine sensitive to these planes for puzzle classification.
  Treating the planes as fixed across counterfactuals therefore preserves
  the trunk's exchange/king features without introducing a confounder.

## Forward-pass cost

The shared trunk encoder runs `1 + K * 4` times per sample in the worst
case (`K = max_promotion_pawns`). At the defaults `K = 4`, that is 17
forward passes per sample. The trunk's deterministic feature builder
(closed-form geometric attack tables) is the dominant per-pass cost; on
positions without own near-promotion pawns the gate is exactly zero, so
the counterfactual passes are constant overhead per batch but
non-contributing.

If wall-clock becomes a problem at scale, two easy optimisations are
available:

1. Drop `K` to 2 (covers the vast majority of real promotion positions —
   even king-and-pawn endings rarely have more than two side-to-move
   pawns on the same near-promotion rank simultaneously).
2. Add a per-sample mask that skips the counterfactual pass entirely
   when `has_promotion_pawn == 0`. This requires per-sample dynamic
   batching, which the current trainer doesn't support, but a small
   gather/scatter wrapper around `self.trunk` can isolate the change.

## Ablations

Allowed `ablation=` values, each documented in `ablations.md`:

| value                       | what it does                                                                |
|-----------------------------|-----------------------------------------------------------------------------|
| `none`                      | Full PFCT primitive.                                                        |
| `copy_baseline_fanout`      | Falsifier A1: replace the four-row fanout with 4 copies of the baseline.   |
| `uniform_attention`         | Disable learned attention (fixed 1/4 weighting across promotion types).    |
| `zero_delta`                | Drive primitive_delta to 0 (effectively i193).                              |
| `force_open_gate`           | Bypass the gate (gate = 1 on positions with at least one near-promotion).  |
| `trunk_only`                | Equivalent to disabling the primitive (`zero_delta`).                      |

`trunk_ablation=` is independent of the PFCT ablation and supports the same
five values exposed by `ExchangeThenKingDualStreamNetwork`
(`none`, `shared_stream_only`, `fixed_half_gate`, `king_only`, `exchange_only`).

## Determinism

- The geometric feature builder is deterministic.
- Near-promotion slot selection is deterministic (sort by file index).
- The counterfactual board construction uses only the piece planes and
  per-slot one-hot indices, so the output is byte-identical for the same
  input.
- The only stochastic component is dropout in the head MLPs; set
  `head_dropout=0.0` in the config to make the primitive deterministic
  end-to-end during eval.

## Encoding caveats

- The simple_18 plane order is `[P, N, B, R, Q, K]` for each colour, so the
  promotion-piece plane offsets in `{Q, R, B, N}` order are
  `(4, 3, 2, 1)` (white) and `(10, 9, 8, 7)` (black).
- Promotion squares: white pawn at rank 7 promotes to rank 8 (plane row 0);
  black pawn at rank 2 promotes to rank 1 (plane row 7). The substitution
  always writes the promoted piece on the same file as the source pawn — no
  diagonal-capture-promotion is enumerated. (A capture-promotion that
  reaches the same file is unreachable in chess; PFCT only enumerates the
  straight-push promotion targets, which is the chess-rule-aligned choice.)
- Existing piece at the promotion square is cleared across all 12 piece
  planes before the promoted piece is set, so capture-promotion contexts
  are handled without producing a multi-piece encoding.

## Trunk reuse and BatchNorm

The PFCT head calls the same `self.trunk` modules (feature builder, stream
encoders) on both the factual and the counterfactual boards. The encoders
contain BatchNorm layers; running them on the counterfactual batch updates
the BN running statistics on what is, by construction, a valid set of
simple_18 boards. This is intentional — the counterfactuals are real chess
positions that the trunk would see in training data anyway, just with the
pawn promoted. If BN drift turns into a problem, switch the trunk to
GroupNorm via `trunk_use_batchnorm: false`.

## What is *not* changed

- The shared trainer is untouched. The model fits the existing
  `model(batch["x"])` -> dict contract.
- The data pipeline is untouched: simple_18 only, no new precomputed
  columns required.
- The i193 baseline is untouched as a class. PFCT only depends on its
  public submodule names (`feature_builder`, `exchange_encoder`,
  `king_encoder`, `exchange_head`, `king_head`, `phase_router`,
  `residual_head`), all of which are stable.

# Architecture

`Material-Phase Low-Rank Adapter Network` is a board-only puzzle-binary classifier that pairs a shared CNN backbone with a stack of low-rank adapters whose per-sample updates are conditioned on a deterministic material/phase summary. The shared weights handle position content; the adapters express phase-conditioned drift without giving the model a path to keep more capacity than its rank allows.

## Input And Material/Phase Summary

- Input is the repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- A deterministic material/phase summary is computed from the input planes only:
  - per-piece counts for the side to move (planes 0..5) and the opponent (planes 6..11)
  - side-to-move bias from plane 12
  - signed material balance using piece values `(P=1, N=3, B=3, R=5, Q=9)`
  - total piece count and a smooth phase coordinate in `[0, 1]` derived from major, minor, and pawn counts
  - castling availability bits from planes 13..16
  - en-passant availability from plane 17
- The summary vector is normalised so each component lives in a comparable range, and a learned `phase_encoder` maps it into a phase embedding used by every adapter generator.

## Shared Backbone

A compact convolutional stem (`BoardConvStem`) processes the full board and produces feature maps that are pooled with mean and max pooling to a backbone embedding of size `2 * channels`. A linear projection lifts that embedding into the adapter hidden width. The backbone weights are not conditioned on the summary.

## Low-Rank Adapter Stack

The hidden state passes through `adapter_blocks` residual blocks. Each block is a `LowRankAdaptedLinear` layer of the form

```
y = W h + b + (1 / r) * B(s) (A(s) h)
```

where `W, b` are shared parameters and `A(s) ∈ R^{r×d}`, `B(s) ∈ R^{d×r}` are produced by linear generators from the phase embedding `s`. The adapter rank `r` is small (default `4`) so the per-sample update is rank-limited by construction. `B(s)` is initialised to zero so training begins at the shared backbone and adapters only earn capacity if material/phase heterogeneity matters. Each block applies LayerNorm, GELU, dropout, and a residual add on top of the adapted output.

## Head And Diagnostics

The classifier consumes the post-adapter hidden state concatenated with the raw material/phase summary and emits one logit for the `puzzle_binary` task (fine labels `0` and `1` map to non-puzzle, fine label `2` maps to puzzle). The forward pass also returns diagnostic tensors of shape `(batch,)`:

- `mean_adapter_norm`, `max_adapter_norm`, and per-block `adapter_block_{i}_norm`: how much the adapter actually moves the hidden state, reported per sample so adapter activity can be sliced by material bucket.
- `backbone_feature_norm` and `phase_summary_norm`: scale of the shared backbone features and of the conditioning signal.
- `material_signed`, `own_material`, `opponent_material`, `own_pawn_count`, `opponent_pawn_count`, `own_minor_count`, `opponent_minor_count`, `own_major_count`, `opponent_major_count`: deterministic material readouts used to bucket evaluations.
- `phase` and `endgame_score`: smooth opening/endgame coordinate (`endgame_score = 1 - phase`).
- `total_piece_count`, `castling_available`, `en_passant_active`, and `side_to_move`: rule/phase scalars from the input encoding.

These diagnostics are aimed at the ablations called out by the source packet: comparing accuracy inside material buckets, watching adapter norms by phase, and checking whether the gain over a `rank_0` baseline survives bucket-conditioned evaluation.

## Implementation Binding

- Registered model name: `material_phase_low_rank_adapter_network`.
- Source implementation: `src/chess_nn_playground/models/material_phase_low_rank_adapter.py`.
- Idea-local wrapper: `ideas/i130_material_phase_low_rank_adapter_network/model.py`.

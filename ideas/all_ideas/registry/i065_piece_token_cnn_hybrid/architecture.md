# Architecture

`Piece-Token CNN Hybrid` combines a dense 8x8 convolutional board encoder
with an explicit occupied-piece token mixer. The two streams are pooled,
concatenated, and (by default) coupled through a multiplicative late
fusion before a small MLP head returns the puzzle logit.

## Streams

The model operates strictly on the simple_18 current-board tensor and
never consumes engine, source, verification, or CRTK metadata.

1. **Board CNN trunk**
   `BoardCNNTrunk` runs a stack of `cnn_blocks` 3x3 conv blocks at the
   default `cnn_width` channels with optional BatchNorm and Dropout2d.
   The trunk preserves the 8x8 spatial layout and emits a feature map
   plus the concatenated mean/max global pool.

2. **Piece-token stream**
   `Simple18PieceTokenExtractor` selects the top
   `max_piece_tokens` (default 32) occupied squares of the simple_18
   piece planes. Each token carries:
   - the 6 piece-type indicators,
   - own/opponent ownership flags relative to the side to move,
   - white/black colour flags,
   - normalized rank/file and side-relative rank/file,
   - the four castling rights as a per-token context vector,
   - the en-passant flag at the token square plus a board-level
     ep-exists flag,
   - the side-to-move flag, and the occupancy weight.

   This yields a 22-dim piece-square feature vector per token. The
   extractor also returns a 20-dim `material_summary` (own/opp piece
   counts, count delta, total count, material balance).

   `PieceTokenMixer` lifts the per-token features through a 2-layer
   MLP into `token_dim` and applies `token_mixer_layers` MaskedSet
   layers. Each layer adds a local MLP residual, then a global
   summary (mean, max, sum) gates a learned set update. The pooled
   token vector concatenates mean, max, and sum across the surviving
   tokens (size `3 * token_dim`).

3. **Late fusion head**
   `CNNTokenFusionHead` projects the CNN and token vectors to
   `fusion_hidden` and multiplies them elementwise to produce an
   explicit interaction term. The interaction is concatenated with the
   raw CNN pool, the raw token pool, and the material summary, then
   passed through a 3-layer MLP returning `num_classes` logits.

## Tensor contract

```text
input:                       (B, 18, 8, 8)
cnn map:                     (B, cnn_width, 8, 8)
cnn pool:                    (B, 2 * cnn_width)
piece tokens:                (B, max_piece_tokens, 22)
token pool:                  (B, 3 * token_dim)
material summary:            (B, 20)
interaction (default):       (B, fusion_hidden)
fused vector:                (B, 2*cnn_width + 3*token_dim + 20 + fusion_hidden)
logits:                      (B,) when num_classes == 1
```

For `num_classes == 1` the head logits are squeezed to `(B,)` so the
puzzle-binary trainer can drive `BCEWithLogitsLoss` directly. The
forward pass returns a dict so the trainer can pull diagnostics:

- `logits`,
- `token_count` (number of selected occupied tokens per board),
- `piece_count` (re-derived total piece count),
- `material_balance` (own minus opponent material, normalized by 39),
- `cnn_energy`, `token_energy`, `cnn_token_interaction`
  (mean square magnitudes for the trunk map, the pooled token vector,
  and the multiplicative interaction term),
- `token_coordinate_energy` (mean magnitude of the rank/file
  coordinate features over the active tokens).

## Falsifier ablations

The model exposes the markdown ablation table via `model.ablation`:

| Ablation | What it removes / changes |
|---|---|
| `"none"` | Default CNN + piece-token hybrid with multiplicative fusion. |
| `"cnn_only_matched"` | Zero out the token stream and material summary so the head sees only the CNN pool; interaction is disabled. |
| `"token_only"` | Zero out the CNN pool so the head depends only on the token + material features. |
| `"no_interaction_fusion"` | Drop the multiplicative interaction term; concatenate streams directly. |
| `"material_token_only"` | Strip the rank/file coordinates and the per-token ep flag from the token features so tokens carry only piece-type + colour + material context. |
| `"shuffle_token_coordinates"` | Roll the rank/file coordinates by one token to break the per-piece spatial binding while preserving the marginal coordinate distribution. |
| `"single_token_layer"` | Reduce `token_mixer_layers` to 1 to test the value of stacked set mixing. |

## Implementation Binding

- Registered model name: `piece_token_cnn_hybrid`.
- Source implementation file:
  `src/chess_nn_playground/models/piece_token_cnn_hybrid.py`
  (defines `Simple18PieceTokenExtractor`, `BoardCNNTrunk`,
  `TokenMixerLayer`, `PieceTokenMixer`, `CNNTokenFusionHead`,
  `PieceTokenCNNHybrid`, plus
  `build_piece_token_cnn_hybrid_from_config`).
- Idea-local wrapper:
  `ideas/all_ideas/registry/i065_piece_token_cnn_hybrid/model.py`
  exports `build_model_from_config` which calls
  `build_piece_token_cnn_hybrid_from_config` after defaulting
  `num_classes` to 1 for the puzzle-binary trainer.
- Inputs are board-only; engine, source, verification, and CRTK
  metadata are never used as model input.

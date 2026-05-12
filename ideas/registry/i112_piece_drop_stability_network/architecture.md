# Architecture

`Piece-Drop Stability Network` is a bespoke architecture that classifies a
position by how much a *shared, compact* convolutional encoder's latent
moves when deterministic piece groups are dropped from the current
board.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / engine metadata is reporting-only and never enters
  the model.

## Pipeline

1. **Original encode.** A compact CNN trunk (`channels` filters,
   `depth` residual conv-norm-act blocks) is followed by a global
   average pool and a linear projection to `latent_dim`, producing
   `z(x) : (B, D)`.
2. **Deterministic piece-drop masks.** For each board, build six
   per-square keep-masks over the 12 piece planes (auxiliary planes
   such as side-to-move and castling rights pass through unchanged):
   - `own_minor`: side-to-move knights and bishops.
   - `own_major`: side-to-move rooks and queens.
   - `opp_minor`: opponent knights and bishops.
   - `opp_major`: opponent rooks and queens.
   - `center`: pieces on the four center squares (d4, e4, d5, e5).
   - `king_neigh`: pieces on a square in the 3x3 neighborhood of either
     king (3x3 max-pool dilation of the king planes).
3. **Masked encodes.** Apply each keep-mask to the piece planes and
   run *the same* shared encoder, giving masked latents
   `z_m : (B, M, D)`.
4. **Stability vector.** Compute per-mask L2 deltas
   `delta_m = ||z(x) - z_m||_2` of shape `(B, M)` and a scale-normalized
   variant `delta_m / (||z(x)|| + eps)`.
5. **Head.** Concatenate `[z(x), delta, delta_ratio]` and feed through
   a `LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(1)`
   classifier returning the puzzle logit, plus diagnostics
   `original_latent`, `masked_latents`, `delta_vectors`, `stability`,
   `stability_ratio`, `original_norm`.

## Tensor Contract

```
input:           (B, 18, 8, 8)
piece_keep:      (B, 12, 8, 8)        per-mask
masked_inputs:   (B, M, 18, 8, 8)     materialised one mask at a time
original_latent: (B, D)
masked_latents:  (B, M, D)
stability:       (B, M)
stability_ratio: (B, M)
logits:          (B,)
```

## Central Ablations (config switches)

| Ablation              | Config knob                              | Effect                                                                |
|-----------------------|------------------------------------------|-----------------------------------------------------------------------|
| `original_only`       | `use_stability=False, use_stability_ratio=False` | Disable both stability heads; head sees only `z(x)`.           |
| `material_masks_only` | `drop_masks: [own_minor, own_major, opp_minor, opp_major]` | Drop center / king-neighborhood spatial masks.       |
| `delta_only`          | `use_original_latent=False`              | Head sees only the stability vector(s), not the original latent.      |
| `random_masks`        | n/a                                       | Reference baseline; not built-in (see trainer notes for procedure).   |

## Implementation Binding

- Registered model name: `piece_drop_stability_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/piece_drop_stability_network.py`
- Idea-local wrapper: `ideas/registry/i112_piece_drop_stability_network/model.py`

The wrapper is a thin adapter over
`build_piece_drop_stability_network_from_config`; it does not touch
`ResearchPacketProbe`. The shared probe wrapper has been removed.

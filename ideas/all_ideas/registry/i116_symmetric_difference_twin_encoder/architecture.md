# Architecture

`Symmetric Difference Twin Encoder` is a bespoke twin-encoder
architecture: one shared convolutional trunk `Phi` is applied to both
the original board and a deterministic safe transform `T(x)` of the
board. Their latents are aligned in the same coordinate frame and
compared by explicit symmetric-difference and intersection features
that the puzzle classifier reads.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / engine metadata is reporting-only and never enters
  the model.

## Safe Deterministic Transform `T`

The model uses **file mirror** as the safe deterministic transform.
For the simple_18 layout, `T` does two things at once:

1. Spatially flips the file axis (`dim=-1`).
2. Permutes channels so that white kingside (plane 13) <-> white
   queenside (plane 14) and black kingside (plane 15) <-> black
   queenside (plane 16) castling planes swap.

The result is a rule-faithful chess position with the same material,
the same side to move, mirrored castling rights, and a mirrored
en-passant file. This is the canonical "safe" symmetry: tactical
content is preserved up to a known coordinate change.

## Pipeline

1. **Twin pass through shared trunk.** The original `x` and the
   transformed `T(x)` are concatenated along the batch dim and pushed
   through one `_SharedBoardTrunk` instance, so the *same* weights and
   BatchNorm statistics see both branches in every step. This shared
   trunk is what makes the architecture a *twin* encoder rather than
   a one-shot CNN. The trunk is `depth` repetitions of
   `Conv2d(_, channels, 3, padding=1) -> Norm -> GELU -> Dropout2d`
   producing a shape-preserving latent `(B, channels, 8, 8)`.
2. **Latent alignment.** With a fully convolutional trunk, the
   inverse `T^{-1}` in latent space is simply a file flip:
   `z_aligned = flip(Phi(T(x)), dim=-1)`. After this step `z` and
   `z_aligned` live in the same spatial frame, so they can be
   compared cell-by-cell.
3. **Symmetric-difference and intersection features.**
   - `preserved = (z + z_aligned) / 2` is the *intersection* in the
     real-vector sense: components that survive the safe transform.
   - `changed = |z - z_aligned|` is the *symmetric difference*:
     components the transform breaks. These are the symmetry-residual
     features the thesis prescribes.
4. **Local fusion.** A small `_DiffFusion` block
   (`Conv2d(2 * channels -> hidden_dim, 3, 1) -> Norm -> GELU ->
   Dropout2d`) consumes the channel-wise concatenation
   `[preserved, changed]` and produces a fused
   `(B, hidden_dim, 8, 8)` map.
5. **Classifier head.** Each of `preserved`, `changed`, and `fused`
   is mean-pooled to `(B, channels)`, `(B, channels)`,
   `(B, hidden_dim)`. They are concatenated and passed through
   `LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(1)`
   to produce one puzzle logit. Pooled features, full feature maps,
   the aligned latents, and several scalar diagnostics
   (`symmetric_difference_energy`, `preserved_energy`,
   `latent_disagreement`, `symmetry_residual`) are returned alongside.

## Tensor Contract

```
input:                       (B, 18, 8, 8)
T(x):                        (B, 18, 8, 8)
z, z_aligned:                (B, channels, 8, 8)
preserved_map, changed_map:  (B, channels, 8, 8)
fused_map:                   (B, hidden_dim, 8, 8)
pooled_preserved:            (B, channels)
pooled_changed:              (B, channels)
pooled_fused:                (B, hidden_dim)
symmetric_difference_energy: (B,)
preserved_energy:            (B,)
latent_disagreement:         (B,)
symmetry_residual:           (B,)
logits:                      (B,)
```

## Why "Symmetric Difference" rather than "Invariance"

Most symmetry-aware networks enforce `Phi(x) = Phi(T(x))`. The thesis
of this idea is the opposite: under a *safe* transform, some evidence
should be preserved (the intersection `preserved`) and other evidence
should change (the symmetric difference `changed`). Forcing
invariance would erase exactly the `changed` signal that this
classifier reads. The head therefore consumes `preserved` and
`changed` as parallel streams instead of collapsing them into a
single invariant representation.

## Central Ablations (config switches)

| Ablation         | Config knob              | Effect                                                                                  |
|------------------|--------------------------|-----------------------------------------------------------------------------------------|
| `narrow_trunk`   | `channels: 32`           | Halves the shared-trunk latent width.                                                   |
| `shallow_trunk`  | `depth: 1`               | Single-conv trunk; tests how much depth the symmetric-difference signal needs.          |
| `wide_head`      | `hidden_dim: 192`        | Doubles the fusion / head width.                                                        |
| `no_dropout`     | `dropout: 0.0`           | Removes regularization on both trunk and head.                                          |
| `no_bn`          | `use_batchnorm: false`   | Replaces BN with GroupNorm(1, ...); useful for tiny batches.                            |

## Implementation Binding

- Registered model name: `symmetric_difference_twin_encoder`
- Source implementation file:
  `src/chess_nn_playground/models/symmetric_difference_twin_encoder.py`
- Idea-local wrapper:
  `ideas/all_ideas/registry/i116_symmetric_difference_twin_encoder/model.py`

The wrapper is a thin adapter over
`build_symmetric_difference_twin_encoder_from_config`; it does not
touch `ResearchPacketProbe`. The shared probe wrapper has been
removed.

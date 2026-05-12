# Architecture

`Tactical Hessian Spectrum Network` is a bespoke `puzzle_binary`
classifier that turns the local-curvature thesis from
`math_thesis.md` into an explicit differentiable computation.

## Thesis recap

A real puzzle is a *sharp local maximum* of tactical evidence under
small legal perturbations of the board. A near-puzzle has high raw
evidence but flatter or less stable local geometry. This architecture
measures local sharpness directly: it builds a scalar tactical-
evidence field `E(x)` over the board, probes its second-order
behaviour along `K` chess-meaningful perturbation directions, and
reads the spectrum of the induced reduced Hessian into the puzzle
classifier.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / engine metadata is reporting-only and never enters
  the model.

## Pipeline

1. **Compact convolutional trunk.** `feats = trunk(x)` runs `depth`
   `Conv2d(_, channels, 3, padding=1) -> Norm -> GELU -> Dropout2d`
   blocks (`Norm` is BatchNorm2d when `use_batchnorm = true`,
   GroupNorm(1, ...) otherwise). The trunk emits a
   `(B, channels, 8, 8)` feature map.
2. **Tactical evidence field.** A `1x1` projection over `feats`
   produces `S(x) ∈ R^(B, 8, 8)`; the scalar evidence is
   `E(x) = sum_{r, f} S(x)[r, f]`.
3. **Perturbation basis.** `K` board-shaped tensors
   `D_k ∈ R^(input_channels, 8, 8)` are seeded from chess-aware
   patterns (color flip, side-to-move flip, rank stripe, file
   stripe). Each `D_k` is unit Frobenius normalised so probes share
   a common scale; a learnable per-direction log-scale (initialised
   to zero) lets the model attenuate or amplify each direction
   without breaking the unit-norm contract.
4. **Reduced Hessian by finite differences.** With step `eps`, we
   evaluate `E` on the variant set
   `{x, x + eps D_k, x - eps D_k, x + eps (D_i + D_j) for i < j}`,
   stacking all variants along the batch and running the trunk
   once. From the variant evidence values we form
   `g_k = (E(x + eps D_k) - E(x - eps D_k)) / (2 eps)`,
   `H_kk = (E(x + eps D_k) + E(x - eps D_k) - 2 E(x)) / eps^2`,
   `H_ij = (E(x + eps (D_i + D_j)) - E(x + eps D_i) - E(x + eps D_j) + E(x)) / eps^2`,
   then symmetrise to obtain a real symmetric `K x K` matrix.
5. **Spectral readout.** `torch.linalg.eigvalsh` produces ascending-
   sorted real eigenvalues. From those we read
   `top_eigenvalue`, `min_eigenvalue`, `spectral_gap`,
   `trace`, `concavity` (negated sum of negative eigenvalues),
   `positive_curvature`, and `spectral_radius`.
6. **Classifier head.** A small MLP
   `LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(1)`
   reads a feature pack assembled from the eigenvalues, the seven
   sharpness scalars, the gradient norm `||g||`, the evidence
   `E(x)`, and pooled trunk features (mean, max, energy) to produce
   one puzzle logit. Sharp negative curvature with large `concavity`
   pushes the position toward the puzzle class; flat or indefinite
   curvature pushes it toward non-puzzle.

## Tensor Contract

```
input x:                          (B, 18, 8, 8)
trunk feats:                      (B, channels, 8, 8)
evidence_field S(x):              (B, 8, 8)
evidence_total E(x):              (B,)
perturbation_directions:          (K, input_channels, 8, 8)
directional_gradient g:           (B, K)
hessian H:                        (B, K, K)
hessian_eigenvalues:              (B, K)
top / min / spectral_gap:         (B,)
trace / concavity:                (B,)
positive_curvature:               (B,)
spectral_radius:                  (B,)
gradient_norm:                    (B,)
trunk_energy:                     (B,)
logits:                           (B,)
```

## Why a Hessian Spectrum Rather Than a Generic Mechanism Probe

The thesis is structural: real puzzles look like sharp local maxima
of tactical evidence; near-puzzles do not. Modelling that explicitly
requires three things — a differentiable evidence scalar over the
board, a basis of legal-style perturbation directions, and a reduced
Hessian whose eigenvalues encode local sharpness. A generic
`ResearchPacketProbe` cannot expose `spectral_gap`, `concavity`, or
`spectral_radius` because it never builds the reduced Hessian to
begin with.

## Material Distinctness

This architecture is materially distinct from:

- The shared `ResearchPacketProbe` scaffold: no evidence scalar, no
  perturbation basis, no reduced Hessian, no eigenvalue readout.
- `KrylovTacticalSubspaceNetwork` (i076): that model builds a chess-
  structured 64x64 operator and reads Arnoldi/Ritz statistics from
  *first*-order operator evolution; this one builds a `K x K`
  reduced Hessian by *second*-order finite differences of an
  evidence scalar under legal perturbations.
- `MatrixPencilGeneralizedSpectrumBottleneck` and other spectrum
  ideas: those compute eigen-features of learned operators on board
  features; this one's spectrum is the curvature spectrum of a
  scalar evidence field, not of a learned linear map.

Removing the perturbation basis, the finite-difference Hessian, or
the eigenvalue readout would change the model's computation in
observable ways and is exactly what the central ablations switch
off.

## Central Ablations (config switches)

| Ablation             | Config knob              | Effect                                                                                  |
|----------------------|--------------------------|-----------------------------------------------------------------------------------------|
| `narrow_trunk`       | `channels: 32`           | Halves the encoder latent width.                                                        |
| `shallow_trunk`      | `depth: 1`               | Single-conv trunk; tests how much depth the evidence field needs.                       |
| `wide_head`          | `hidden_dim: 192`        | Doubles the head width.                                                                 |
| `coarse_step`        | `eps: 1.0`               | Larger finite-difference step probes a wider neighbourhood.                             |
| `fine_step`          | `eps: 0.1`               | Smaller step approaches the differential Hessian.                                       |
| `few_directions`     | `num_directions: 2`      | Smallest non-degenerate Hessian; tests whether 4 directions are needed.                 |
| `extra_directions`   | `num_directions: 6`      | Adds rank-stripe padding directions to the basis.                                       |
| `no_dropout`         | `dropout: 0.0`           | Removes regularization on encoder and head.                                             |
| `no_bn`              | `use_batchnorm: false`   | Replaces BN with GroupNorm(1, ...); useful for tiny batches.                            |

## Implementation Binding

- Registered model name: `tactical_hessian_spectrum_network`
- Source implementation file:
  `src/chess_nn_playground/models/trunk/tactical_hessian_spectrum_network.py`
- Idea-local wrapper:
  `ideas/registry/i199_tactical_hessian_spectrum_network/model.py`

The wrapper is a thin adapter over
`build_tactical_hessian_spectrum_network_from_config`; it does not
touch `ResearchPacketProbe`. The shared probe wrapper has been
removed.

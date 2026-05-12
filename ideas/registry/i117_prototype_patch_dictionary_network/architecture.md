# Architecture

`Prototype Patch Dictionary Network` is a bespoke patch-dictionary
classifier: each board is decomposed into per-square patch embeddings
that are decoded against a small learned dictionary of motif
prototypes. The puzzle classifier reads the resulting **motif
assignment map**, **reconstruction residual**, and **prototype
activation histogram** -- exactly the diagnostics the thesis prescribes.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / engine metadata is reporting-only and never enters
  the model.

## Patch Embedding

A small `_PatchEncoder` (a stack of `Conv2d -> Norm -> GELU ->
Dropout2d` blocks, depth `depth`, first kernel `patch_kernel`,
subsequent kernels `3`) maps the input to a shape-preserving feature
map `(B, patch_dim, 8, 8)`. The first kernel sets the local
neighborhood radius; with `patch_kernel: 3` each square's embedding
sees the 3x3 stencil "the square plus its 8 neighbours" that is the
natural unit for tactical motifs.

The map is reshaped to per-square patches `p_{b, s} in R^{patch_dim}`
with `s = 0..63`.

## Learned Patch Dictionary

The model holds a dictionary `D = [d_1, ..., d_K] in R^{K x patch_dim}`
of motif prototypes (`K = num_prototypes`). The rows are L2-normalised
before the cosine-similarity step so similarity scores are bounded.
The dictionary is initialised orthogonally so prototypes start
well-separated.

## Soft Motif Assignment

For every square `s` the encoder produces a soft assignment over the
`K` prototypes:

```
similarity_{b, s, k} = <normalize(p_{b, s}), normalize(d_k)>
alpha_{b, s, k}      = softmax_k( similarity_{b, s, k} / tau )
```

`tau` is a learned positive temperature (parameterised as `exp(log_tau)`
so it stays positive without clamping). The full `(B, K, 8, 8)` map
`alpha` is the **motif assignment map**; its argmax over `k` gives the
top-1 motif id per square (`top1_motif_map`).

## Reconstruction and Residual

The patch is reconstructed as the convex combination
`p_hat_{b, s} = sum_k alpha_{b, s, k} * d_k`. The classifier head sees
the **reconstruction residual** `r_{b, s} = p_{b, s} - p_hat_{b, s}`
and its per-square energy `||r_{b, s}||^2`. We also expose
`residual_per_prototype`, the soft-assignment-weighted residual mass
attributed to each prototype direction.

## Prototype Activation Histogram

The **prototype activation histogram** is the spatial mass of the soft
assignment per prototype:

```
prototype_histogram_{b, k} = sum_s alpha_{b, s, k} / 64
```

It sums to 1 across prototypes and tells the head "how strongly each
motif is represented on this board". `prototype_entropy` summarises
how uniform that distribution is.

## Classifier Head

The head receives the concatenation of the prototype histogram, the
pooled residual magnitude per dimension, and pooled patch /
reconstruction embeddings:

```
head_input = concat(
    prototype_histogram,            # (B, K)
    pooled_residual = mean_s |r|,   # (B, patch_dim)
    pooled_patch    = mean_s p,     # (B, patch_dim)
    pooled_recon    = mean_s p_hat, # (B, patch_dim)
)
LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(1)
```

This produces one puzzle logit. All assignment maps, residual maps,
reconstruction maps, and scalar diagnostics are returned alongside.

## Tensor Contract

```
input:                          (B, 18, 8, 8)
patch_map:                      (B, patch_dim, 8, 8)
patches:                        (B, 64, patch_dim)
soft_assignment:                (B, 64, K)
assignment_map:                 (B, K, 8, 8)
top1_motif_map:                 (B, 8, 8)            (long)
reconstruction_map:             (B, patch_dim, 8, 8)
residual_map:                   (B, patch_dim, 8, 8)
prototype_histogram:            (B, K)
residual_per_prototype:         (B, K)
residual_energy_per_square:     (B, 8, 8)
residual_energy:                (B,)
prototype_entropy:              (B,)
temperature:                    (B,)
pooled_patch / pooled_recon /
pooled_residual:                (B, patch_dim)
logits:                         (B,)
```

## Why "Patch Dictionary" rather than a generic CNN

A vanilla CNN can also produce a `(B, K, 8, 8)` channel map, but that
map is not committed to a single set of prototypes the head can
reconstruct from. The dictionary `D` is *the same* set of vectors
used both to compute assignments (cosine softmax) and to reconstruct
the patches (convex combination). That coupling is what makes the
assignment and the reconstruction residual interpretable as motif use
and reconstruction failure, respectively. Replacing the
softmax-cosine assignment with a generic `Conv2d`, or untying the
"assignment direction" and the "reconstruction direction", deletes the
diagnostics the classifier reads.

## Central Ablations (config switches)

| Ablation              | Config knob              | Effect                                                                                                |
|-----------------------|--------------------------|-------------------------------------------------------------------------------------------------------|
| `narrow_patches`      | `channels: 32`           | Halves the patch-embedding dimension `patch_dim`.                                                     |
| `shallow_encoder`     | `depth: 1`               | Single-layer patch encoder; tests whether motifs need any non-linear context beyond the first conv.   |
| `wide_head`           | `hidden_dim: 192`        | Doubles the head width.                                                                               |
| `tiny_dictionary`     | `num_prototypes: 8`      | Forces the model to compress motifs into a coarser dictionary.                                        |
| `large_dictionary`    | `num_prototypes: 64`     | Increases motif granularity; tests whether the histogram saturates.                                   |
| `square_only_patches` | `patch_kernel: 1`        | Removes the 3x3 neighborhood from the first conv so patches see only the square itself.               |
| `no_dropout`          | `dropout: 0.0`           | Removes regularization on the patch encoder and head.                                                 |
| `no_bn`               | `use_batchnorm: false`   | Replaces BN with GroupNorm(1, ...); useful for tiny batches.                                          |

## Implementation Binding

- Registered model name: `prototype_patch_dictionary_network`
- Source implementation file:
  `src/chess_nn_playground/models/prototype_patch_dictionary_network.py`
- Idea-local wrapper:
  `ideas/registry/i117_prototype_patch_dictionary_network/model.py`

The wrapper is a thin adapter over
`build_prototype_patch_dictionary_network_from_config`; it does not
touch `ResearchPacketProbe`. The shared probe wrapper has been
removed.

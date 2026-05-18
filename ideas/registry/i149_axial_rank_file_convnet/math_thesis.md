# Math Thesis

Axial Rank-File ConvNet

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `1`.

Working thesis: Use ordinary convolutions, but factor long-range board mixing into alternating `8`-length rank and file convolutions. Every square sees same-rank and same-file context cheaply, with no attention, ray solver, or multi-resolution pyramid.

## Operator description

Let `z in R^{B, C, 8, 8}` denote the working feature map after a 3x3 conv stem on the simple_18 board. The block operator is

```
A(z) = z + Dropout(
        GELU(BN(K_R conv z))
      + GELU(BN(K_F conv z))
      + GELU(BN(K_L conv z))
    )
```

where `K_R` has spatial shape `(1, 8)` and reaches every file index, `K_F` has spatial shape `(8, 1)` and reaches every rank index, and `K_L` is the local 3x3 mixer. Padded outputs of the 1D convs are truncated back to `8x8`. The full trunk is `A^L(stem(x))` for depth `L`.

This factorisation is the standard axial decomposition of a long-range linear operator: every per-square update is a sum of (a) a rank-local affine map of the rank's 8-vector, (b) a file-local affine map of the file's 8-vector, and (c) a 3x3 local correction. Compared with a full `8x8` convolution which would need `64 C^2` parameters per layer, the axial decomposition uses only `(8 + 8 + 9) C^2 = 25 C^2`.

## Pooled features

The head reads three pools of the trunk output `h = A^L(stem(x))`:

```
pool_R(h) = [mean_w(h); max_w(h)]       in R^{B, 16 C}
pool_F(h) = [mean_h(h); max_h(h)]       in R^{B, 16 C}
pool_G(h) = [mean_{h, w}(h); max_{h, w}(h)] in R^{B, 2 C}
```

Concatenated, `phi(x) = [pool_R; pool_F; pool_G] in R^{B, 34 C}`. The classifier is a `LayerNorm + MLP -> num_classes` map on `phi(x)`.

## Decision rule

`phi(x)` is normalised by a `LayerNorm` and fed through

```
phi(x) -> Linear(34 C, H) -> GELU -> Dropout? -> Linear(H, H/2) -> GELU -> Dropout? -> Linear(H/2, num_classes)
```

producing one logit per board for the BCE-with-logits puzzle head.

## Falsification path

The central falsifier is `local_only`, which zeros both axial branches so only the local 3x3 mixer survives. If the local-only model matches the full model, axial 1D mixing is not what is helping. `rank_only` and `file_only` ablate one axial branch at a time to test whether either direction alone suffices, or whether both directions are needed. `no_residual` drops the residual skip to test whether the residual-stream structure is required. `single_block` collapses the trunk to a single axial block to test whether deeper axial stacks help.

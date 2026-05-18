# Architecture

`Axial Rank-File ConvNet` is a bespoke axial CNN for the simple_18 board. It factors long-range board mixing into alternating 8-length rank and file 1D convolutions, with a local 3x3 residual mixer per block. Every square gets cheap same-rank and same-file context without any attention layer, ray solver, or multi-resolution pyramid.

## Implementation Binding

- Registered model name: `axial_rank_file_convnet`
- Source implementation file: `src/chess_nn_playground/models/trunk/axial_rank_file_convnet.py`
- Idea-local wrapper: `ideas/registry/i149_axial_rank_file_convnet/model.py`

## Modules

`AxialRankFileStem` is a Conv2d 3x3 (+ optional BatchNorm) + GELU that maps the `simple_18` planes to `channels` working channels.

`AxialRankFileBlock` is the core axial block. It runs three parallel branches on the same input feature map and sums their outputs into the residual stream:

- `rank_conv`: `Conv2d` with kernel size `(1, 8)` and zero-padding `(0, 4)`, truncated back to `8x8`. This is a rank-wise 1D convolution: every square sees all 8 squares on its own rank.
- `file_conv`: `Conv2d` with kernel size `(8, 1)` and zero-padding `(4, 0)`, truncated back to `8x8`. This is a file-wise 1D convolution: every square sees all 8 squares on its own file.
- `local_conv`: `Conv2d` with kernel size `3` and padding `1`, the standard local 3x3 mixer.

Each branch is followed by BatchNorm (optional) and GELU. Their sum is passed through Dropout2d (optional) and added to the residual stream: `x_out = x + Dropout(GELU(BN(rank_conv(x))) + GELU(BN(file_conv(x))) + GELU(BN(local_conv(x))))`.

`AxialRankFileHead` is the classifier. It pools the trunk output with three concatenated pools:

- Rank pool: `[mean_w(z); max_w(z)]` flattened to `R^{B, 16 * C}`.
- File pool: `[mean_h(z); max_h(z)]` flattened to `R^{B, 16 * C}`.
- Global pool: `[mean_{h, w}(z); max_{h, w}(z)]` to `R^{B, 2 * C}`.

The total pooled dimension is `34 * C`. A `LayerNorm + (Linear + GELU + Dropout) x 2 + Linear(num_classes)` MLP produces one BCE-compatible logit for `puzzle_binary`.

## Ablation modes

`AxialRankFileConvNet.ABLATIONS` enumerates the testable variants:

- `none` (default): full block — rank + file + local branches active.
- `local_only`: zero the rank and file branches; only the 3x3 local mixer runs. Tests whether axial 1D mixing matters at all.
- `rank_only`: zero the file and local branches; only the rank-wise 1D conv runs. Tests whether one axial direction alone suffices.
- `file_only`: zero the rank and local branches; only the file-wise 1D conv runs. Symmetric control to `rank_only`.
- `no_residual`: replace the residual update `x + update` with `update` itself. Tests whether the residual skip matters.
- `single_block`: collapse the trunk to a single axial block regardless of the configured depth. Tests whether deeper axial stacks help.

## Diagnostics

`forward(x)` returns a dict containing:

- `logits`: shape `(B,)` (or `(B, num_classes)` for non-binary configs), BCE-compatible for the one-logit `puzzle_binary` head.
- `prob`: sigmoid of the puzzle logit (softmax when `num_classes > 1`).
- `trunk_energy`: mean square of the trunk output.
- `rank_energy`, `file_energy`, `local_energy`: mean square of each branch's contribution averaged across blocks.
- `axial_balance`: `(rank + file) / (rank + file + local + eps)` per sample.
- `rank_file_imbalance`: per-sample absolute difference between `rank_energy` and `file_energy`.
- `rank_pool_norm`, `file_pool_norm`, `global_pool_norm`: pooled feature norms feeding the head.
- `piece_density`: side-summed piece-plane occupancy averaged over the board.
- `mechanism_energy`: `rank_energy + file_energy + local_energy`; operationalises the axial mechanism.
- `proposal_profile_strength`: per-sample max over the three branch energies.
- `proposal_keyword_count`: scalar count of active branches (3).
- `axial_rank_file_ablation`: integer code identifying the active ablation mode.
- `axial_rank_file_block_count`: scalar reporting the active block count.

## Contract

- Input: `(B, 18, 8, 8)` board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Model shapes: trunk `z ∈ R^{B, C, 8, 8}`, pooled head input `R^{B, 34 * C}`.
- The puzzle decision flows only through the pooled axial feature vector; the axial structure is enforced architecturally by the per-branch 1D kernels covering the entire 8-length axis.
- Only the conv weights, BatchNorms, LayerNorm, and head linears are trainable.

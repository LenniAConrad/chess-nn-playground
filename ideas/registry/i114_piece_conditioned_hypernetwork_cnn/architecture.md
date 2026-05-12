# Architecture

`Piece-Conditioned Hypernetwork CNN` is a bespoke per-sample CNN whose
depthwise filters and channel gates are produced by a small
hypernetwork conditioned on a deterministic piece-inventory summary.
There are no static depthwise kernels: every block applies a
`(B, C, K, K)` weight tensor that the hypernetwork emits per sample.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / engine metadata is reporting-only and never enters
  the model.

## Pipeline

1. **Piece-plane embedding.** A 1x1 convolution lifts the 18 input
   planes to `channels` channels; this performs no spatial mixing.
2. **Inventory summary.** A deterministic `(B, 27)` summary is built
   from the raw board planes: white/black piece-type counts (12),
   per-type material deltas (6), means of the 6 state planes,
   total occupancy, total material delta, and a minor-piece
   imbalance score. The summary is the *only* signal fed to the
   hypernetwork.
3. **Summary encoder.** A two-layer MLP with LayerNorm maps the
   summary to a shared `(B, hyper_hidden)` embedding `e(x)`.
4. **Per-block hyper-heads.** For each of `depth` blocks, two
   linear heads emit per-sample weights from `e(x)`:
   - `gates(x) ∈ R^C`: per-channel sigmoid gates initialized so an
     untrained network starts with most channels strongly active.
   - `kernels(x) ∈ R^{C × 1 × K × K}`: per-channel depthwise kernel
     weights initialized to a centered identity-like stencil so the
     untrained block behaves like a near-passthrough residual.
5. **Hyper-conditioned residual block.** Given a feature stream
   `x : (B, C, 8, 8)` and the predicted `(gates, kernels)`:
   1. BatchNorm over the channel axis.
   2. Per-sample depthwise `K × K` convolution implemented via
      grouped `conv2d` with `groups = B * C` so each sample's
      channels each apply their own predicted kernel.
   3. Static pointwise 1x1 conv (`channels` in / out).
   4. GELU activation, optional `Dropout2d`.
   5. Multiplication by `gates` broadcast across the spatial axes.
   6. Residual add to the input.
6. **Pool and head.** Mean-pool over the spatial axes, LayerNorm
   the pooled feature, then a `Linear(hidden_dim) -> GELU -> Dropout
   -> Linear(1)` head returns the puzzle logit. Per-block gate
   statistics, kernel norms, block-output energies, and the
   inventory-summary vector are returned as diagnostics.

## Tensor Contract

```
input:                       (B, 18, 8, 8)
inventory_summary:           (B, 27)
embedded:                    (B, C, 8, 8)
per-block gates:             (B, C)
per-block kernels:           (B, C, 1, K, K)
gate_mean_per_block:         (B, depth)
gate_entropy_per_block:      (B, depth)
kernel_norm_per_block:       (B, depth)
block_energy_per_block:      (B, depth)
pooled_features:             (B, C)
logits:                      (B,)
material_delta:              (B,)
```

## Central Ablations (config switches)

| Ablation        | Config knob                              | Effect                                                                |
|-----------------|------------------------------------------|-----------------------------------------------------------------------|
| `shallow_depth` | `depth: 1`                               | Single conditioned block; tests whether one adapted pass suffices.    |
| `wide_hyper`    | `hyper_hidden: 192`                      | Increase the hypernetwork's hidden width to test conditioning capacity. |
| `narrow_trunk`  | `channels: 32`                           | Halve the trunk channels to test parameter-budget sensitivity.        |
| `larger_kernel` | `kernel_size: 5`                         | Use a 5x5 predicted depthwise kernel instead of 3x3.                  |

## Implementation Binding

- Registered model name: `piece_conditioned_hypernetwork_cnn`
- Source implementation file: `src/chess_nn_playground/models/piece_conditioned_hypernetwork_cnn.py`
- Idea-local wrapper: `ideas/registry/i114_piece_conditioned_hypernetwork_cnn/model.py`

The wrapper is a thin adapter over
`build_piece_conditioned_hypernetwork_cnn_from_config`; it does not
touch `ResearchPacketProbe`. The shared probe wrapper has been
removed.

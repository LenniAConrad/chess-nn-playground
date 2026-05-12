# Architecture

`Spline Board Surface Network` is a board-only puzzle_binary classifier that
fits a smooth, low-degree tensor-product spline surface to each piece plane on
the 8x8 grid and reads its smooth coefficients, residual energies, and a
compact residual-map summary.  It is *not* a CNN: there is no convolutional
trunk over the original board planes; the only convolution in the model is a
1x1 mixer applied to residuals.

## Pipeline

1. A precomputed tensor-product Bernstein basis ``basis : (64, K)`` over the
   8x8 grid is built once at construction time, where ``K =
   spline_basis_size**2``.  Its Moore-Penrose pseudoinverse
   ``basis_pinv : (K, 64)`` is also cached as a non-trainable buffer.
2. For each of the ``input_channels = 18`` piece planes the smooth surface
   coefficients are obtained by least-squares projection,
   ``coeffs = basis_pinv @ plane_flat``.  The smooth reconstruction is
   ``reconstruction = basis @ coeffs``, and the residual map is
   ``residuals = plane - reconstruction``.
3. The classifier head receives, in this order:
   * **smooth coefficients** -- ``coefficients : (B, 18, K)`` describing the
     low-degree surface fit per plane, flattened to ``(B, 18*K)``;
   * **residual energies** -- ``residual_energy : (B, 18)`` = squared
     Frobenius norm of each plane's residual map (the sharp / "non-smooth"
     mass per piece plane);
   * **residual map summary** -- a 1x1 convolution mixes the 18 residual
     planes channel-wise into ``residual_summary_channels`` feature maps,
     normalises with ``LayerNorm`` and produces both a mean and a max pool
     over the 8x8 grid (``residual_summary_mean`` and
     ``residual_summary_max``, each ``(B, residual_summary_channels)``).
4. The head is ``LayerNorm -> Linear -> GELU -> Dropout -> Linear`` and
   emits a single puzzle logit ``logits : (B,)``.  All diagnostic tensors
   are exposed alongside the logit so ablations and reports can read them
   without a second forward pass.

## Tensor Contract

```text
input:                 (B, 18, 8, 8)
basis:                 (64, K)              (non-trainable buffer, K = spline_basis_size**2)
basis_pinv:            (K, 64)              (non-trainable buffer)
coefficients:          (B, 18, K)
reconstruction:        (B, 18, 8, 8)
residuals:             (B, 18, 8, 8)
residual_energy:       (B, 18)
residual_summary_mean: (B, residual_summary_channels)
residual_summary_max:  (B, residual_summary_channels)
logits:                (B,)
```

The head input dimensionality is
``18*K + 18 + 2*residual_summary_channels``.

## Why this is not a shared probe

There are no proposal-profile diagnostics, no mechanism-family embeddings, and
no shared `ResearchPacketProbe` code.  The signal that reaches the head is
exactly the smooth-surface decomposition prescribed by ``math_thesis.md`` --
spline coefficients, residual energies, and a residual map summary --
supplemented only by a single 1x1 channel mixer on the residual maps to give
the head a compact view of where the smooth fit failed.  Ablations on
``spline_basis_size`` and ``residual_summary_channels`` map directly to the
central design knobs in the source packet, and ablations that hide individual
components (drop the coefficients, drop the residual energies, or drop the
residual summary) are well-defined operations on this code path.

## Implementation Binding

- Registered model name: `spline_board_surface_network`.
- Source implementation file:
  `src/chess_nn_playground/models/spline_board_surface_network.py`.
- Idea-local wrapper:
  `ideas/all_ideas/registry/i110_spline_board_surface_network/model.py` (a thin
  `build_model_from_config` over
  `build_spline_board_surface_network_from_config`; no
  `ResearchPacketProbe` is involved).

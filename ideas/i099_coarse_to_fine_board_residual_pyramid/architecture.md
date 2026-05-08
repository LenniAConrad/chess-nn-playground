# Architecture

`Coarse-to-Fine Board Residual Pyramid` is a bespoke PyTorch puzzle-binary classifier that builds a two-level residual pyramid over the simple_18 board feature map. It predicts each finer scale from the coarser scale, then classifies from a *coarse stream* (deterministic board summary plus pooled coarsest-scale features) jointly with a *residual stream* (pooled per-scale residuals plus residual-energy statistics) so that the residual head only contributes signal that the coarse reconstructions could not explain.

## Implementation Binding

- Registered model name: `coarse_to_fine_board_residual_pyramid`
- Source implementation file: `src/chess_nn_playground/models/coarse_to_fine_residual_pyramid.py`
- Idea-local wrapper: `ideas/i099_coarse_to_fine_board_residual_pyramid/model.py`

## Modules

`BoardSummary` is a deterministic, parameter-free 18-D board summary computed on the simple_18 planes: 12-D per-piece occupancy counts, side-to-move plane mean, signed material balance, occupancy count, rank/file imbalance and central pressure. The summary is concatenated with the coarsest-scale pooled features so the coarse stream cannot leak fine-scale detail.

`ConvNormGelu` is the basic compute unit: `Conv2d(k×k) → BatchNorm2d (or Identity) → GELU → optional Dropout2d`. `PyramidResidualBlock` stacks two such convolutions with a GELU-activated identity skip and is used at the stem and to refine each residual.

`CoarseToFineBoardResidualPyramid.stem` augments the raw board with two normalized coordinate planes `(rank, file) ∈ [-1, 1]^2`, then runs `ConvNormGelu(C+2 → channels)` followed by `depth × PyramidResidualBlock(channels)` to produce the finest-scale feature map `f_8 ∈ R^{channels × 8 × 8}`. Two `ConvNormGelu` projections combined with `avg_pool2d` produce coarser feature maps `f_4 ∈ R^{channels × 4 × 4}` and `f_2 ∈ R^{channels × 2 × 2}`.

The two decoders `decode2_to4` and `decode4_to8` are `ConvNormGelu(channels → channels) → Conv2d(channels → channels, 1×1)` blocks. They are followed by a bilinear `interpolate` to the next finer spatial size to give predictions `pred_4` and `pred_8` of the next-finer feature maps. The residuals `r_4 = refine_4(f_4 − pred_4)` and `r_8 = refine_8(f_8 − pred_8)` are passed through a `PyramidResidualBlock` each, and the explained map at every scale is `expl = pred + residual_scale · residual`.

The `coarse_head` is `Linear(2·channels + 18 → hidden_dim) → LayerNorm → GELU` over the concatenation of `(mean, max)`-pooled `f_2` and the deterministic board summary. The `residual_head` is `Linear(4·channels + 10 → hidden_dim) → LayerNorm → GELU → optional Dropout` over `(mean, max)`-pooled `r_4` and `r_8` together with ten residual-energy diagnostics: per-scale `L^1`, `L^2`, `L^∞` of `r_4` and `r_8`, the coarse `L^2` of `f_2`, the unexplained-energy ratio `||r_8||_2 / (||expl_8||_2 + ||r_8||_2)`, the residual-gain ratio `||r_8||_2 / ||r_4||_2`, and the sparsity proxy `||r_8||_∞ / ||r_8||_1`. The puzzle decision is `s = Linear(2·hidden_dim → 1)([head_coarse, head_residual])`.

`residual_scale` is a hyperparameter `α ≥ 0` that multiplies `r_4` and `r_8` when forming `expl_4` and `expl_8`. When `α = 0` the model collapses to a coarse-only classifier (the residual diagnostics still flow into the residual head, so the residual stream remains testable in isolation but the explained pyramid contains no learned residual detail).

## Contract

- Input: `(B, C, 8, 8)` simple_18 board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus diagnostics: `coarse_l2`, `explained_l2`, `residual4_l1`, `residual4_l2`, `residual4_max`, `residual8_l1`, `residual8_l2`, `residual8_max`, `unexplained_ratio`, `residual_gain`, `detail_concentration`, `residual_alignment`, and the pass-through summary scalars (`material_balance`, `occupancy_count`, `rank_file_imbalance`, `center_pressure`).
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Ablation: setting `residual_scale=0.0` drops the learned residual contribution from `expl_8` while leaving the residual diagnostics intact, exposing the coarse-only baseline.

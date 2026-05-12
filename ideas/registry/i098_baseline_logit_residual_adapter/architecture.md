# Architecture

`Baseline Logit Residual Adapter` is a bespoke PyTorch puzzle-binary classifier that decomposes its prediction into an explicit `simple_cnn`-shaped *baseline logit* plus a learned *residual adapter correction*. The residual branch is conditioned on the baseline logit, the baseline pooled latent, and a deterministic board summary, so its head only adds signal that is *not* already explained by the baseline.

## Implementation Binding

- Registered model name: `baseline_logit_residual_adapter`
- Source implementation file: `src/chess_nn_playground/models/trunk/baseline_logit_residual_adapter.py`
- Idea-local wrapper: `ideas/registry/i098_baseline_logit_residual_adapter/model.py`

## Modules

`BaselineLogitBranch` is the simple-CNN-shaped reference predictor: `depth` repeats of `Conv2d(3x3) → BatchNorm → GELU → optional Dropout2d`, followed by global-mean pooling into a `channels`-dim latent and a `Linear(channels → 1)` baseline logit head. It exposes its feature map, pooled latent `z_b ∈ R^{channels}`, and baseline logit `s_b ∈ R` so the adapter can condition on all three.

`Simple18BoardSummary` is a deterministic, parameter-free 26-D board summary computed on the simple_18 planes: per-piece occupancy counts, side-to-move, signed material balance, total material, occupancy count, rank/file/joint occupancy imbalance, central pressure, king-ring pressure, and a 5-D auxiliary aggregate over the residual planes. The summary is passed to the adapter as deterministic baseline-conditioning context `m`.

`FiLMResidualAdapter` is the learned residual branch. Its inputs are the raw board planes augmented with two coordinate planes `(rank, file) ∈ [-1, 1]^2`. The branch path is:

1. `input_projection`: `Conv2d(C+2 → A, 1x1) → BatchNorm → GELU` produces an adapter feature map of width `A` (default `max(8, channels // 2)`).
2. `baseline_projection`: `Conv2d(channels → A, 1x1)` projects the baseline feature map (optionally detached) into the adapter width and is added to the projected board.
3. A FiLM modulation reads the joint condition `c = [z_b, s_b, m] ∈ R^{channels + 1 + 26}` through a single `Linear(condition_dim → 2A)` to produce `(γ, β)`. The adapter map is modulated as `(1 + 0.25 tanh γ) · h + β`, broadcast over the spatial axes — a bounded multiplicative gate plus an additive bias.
4. `ResidualAdapterBlock × max(1, depth)`: each block is `Conv2d(A → A, 3x3, depthwise) → BatchNorm → GELU → Conv2d(A → A, 1x1) → BatchNorm → optional Dropout2d`, added to the input via a GELU-activated skip — a depthwise-separable residual stack.
5. Pool: spatial mean and max are concatenated into a `2A`-d adapter pooled vector.
6. `residual_head`: `Linear(2A + condition_dim → hidden_dim) → LayerNorm → GELU → optional Dropout → Linear(hidden_dim → 1)` returns the residual logit `s_r`.
7. `gate`: `Linear(condition_dim → hidden_dim) → GELU → Linear(hidden_dim → 1) → sigmoid` returns a per-example scalar gate `g ∈ (0, 1)` so the residual correction can shrink to zero whenever the conditioning context already explains the example.

The final logit is `s = s_b + α · g · s_r` with `α = residual_scale` (default `1.0`). When `detach_baseline_context=True` (default) the baseline branch is the only path that shapes `s_b`; the adapter cannot rewrite the baseline through gradients on the conditioning context, so `s_r` is forced to encode information *residual* to the baseline.

## Contract

- Input: `(B, C, 8, 8)` simple_18 board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus diagnostics: `baseline_logit`, `residual_logit`, `adapter_correction`, `residual_gate`, `baseline_probability`, `residual_to_baseline_ratio`, `baseline_latent_norm`, `adapter_feature_norm`, `adapter_field_energy`, and the pass-through summary scalars (`material_balance`, `material_total`, `occupancy_count`, `rank_file_imbalance`, `center_pressure`, `king_ring_pressure`).
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- The puzzle decision is `s = s_b + α · g · s_r`, so the baseline-only ablation is recovered exactly by setting `residual_scale=0.0`, and the residual branch is testable in isolation by reading `residual_logit` / `adapter_correction`.

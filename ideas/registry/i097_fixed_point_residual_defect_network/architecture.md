# Architecture

`Fixed-Point Residual Defect Network` realises the source packet's fixed-point / residual-defect thesis as a bespoke PyTorch model for the repository's `puzzle_binary` task. The classifier consumes a damped fixed-point iteration's defect trajectory rather than the final latent alone, so the head reads the position primarily through the residuals of an unrolled update operator.

## Implementation Binding

- Registered model name: `fixed_point_residual_defect_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/fixed_point_residual.py`
- Idea-local wrapper: `ideas/registry/i097_fixed_point_residual_defect_network/model.py`

## Modules

`BoardFixedPointEncoder` is a compact convolutional trunk over the simple_18 board planes, optionally augmented with two coordinate planes that broadcast `(rank, file) ∈ [-1, 1]^2`. The trunk is `Conv2d → GroupNorm → GELU` repeated `depth` times, mean-and-max pooled into a `2 * channels`-d global descriptor, and projected by two parallel `Linear → LayerNorm → GELU` heads into the initial latent `h_0 ∈ R^{latent_dim}` and the board-conditioning embedding `c ∈ R^{board_embed_dim}` consumed by the update operator.

`ResidualUpdateBlock` is the learned fixed-point map `T_φ : R^{latent_dim + board_embed_dim} → R^{latent_dim}`: `Linear → LayerNorm → GELU → Linear → GELU → Linear`. The model carries one *shared* `T_φ` and a `nn.ModuleList` of `K` *untied* update blocks; the active variant is selected at runtime by the mode argument (`untied_residual_blocks` switches to the untied stack, the default uses the shared block, and `random_update_operator` freezes the shared block at initialisation).

The damped fixed-point iteration is

```
h_{k+1} = h_k + alpha * (T_phi(h_k, c) - h_k)
r_k     = T_phi(h_k, c) - h_k        for k = 0, ..., K-1
r_K     = T_phi(h_K, c) - h_K        (terminal defect)
```

so the defect trajectory `(r_0, ..., r_{K-1})` is the central object, and `r_K` is exposed separately as the *final defect* for the path-summary block.

`DefectTrajectoryStats` is the bottleneck. Given the latent path `(h_0, ..., h_K)`, the residual stack `(r_0, ..., r_{K-1})`, and the terminal defect `r_K`, it forms a permutation-equivariant statistic of the defect trajectory:

- `r_l2[k] = ||r_k||_2`, `r_l1[k] = mean |r_k|` per step.
- `cosine[k] = cos(r_k, r_{k-1})` with `cosine[0] = 0` and `oscillation[k] = 1 - cosine[k]`.
- `contraction[k] = ||r_k|| / max(||r_{k-1}||, eps)` with `contraction[0] = 1`.
- `signed_delta[k] = ||r_k|| - ||r_{k-1}||` with `signed_delta[0] = 0`.
- A learned `projection_dim x latent_dim` row-normalised matrix `P` produces `r_proj[k] = P r_k`, exposing per-step components of the residual along learned axes.
- A global `defect_stats` block: total path length `sum_k r_l2[k]`, mean / max / terminal `r_l2`, `||r_K||_2`, `||r_K||_1`, mean contraction ratio, and mean oscillation.

The full readout concatenates the per-step `(r_l2, r_l1, cos, contraction, signed_delta, r_proj)` features with the global `defect_stats` block and (optionally) the final latent `h_K`. The head is a `LayerNorm → Linear → GELU → Dropout → Linear → GELU → Linear(1)` MLP returning a single puzzle logit. Two parallel heads (`norm_head` and `final_head`) are pre-built so the `defect_norm_only` and `final_latent_only` ablation modes can read shape-specific feature blocks without rebuilding the model.

`FixedPointResidualDefectNetwork` wires the trunk together: encoder → `K`-step damped fixed-point iteration → defect trajectory stats → mode-specific head. The puzzle decision flows only through `ψ(x) = stats(h_path, r_path, r_K)`, so the residual-defect bottleneck is enforced architecturally rather than declared in documentation.

## Modes

The `mode` argument selects the active variant; the architecture exposes a numeric `fixed_point_mode` code in the diagnostics so ablation harnesses can attach the active branch to each prediction.

- `none` / `fixed_point` (default): the full residual-defect trajectory readout described above.
- `final_latent_only`: the head reads only the final latent `h_K` (via `final_head`). Tests whether the residual-defect channels add anything beyond the final latent.
- `defect_norm_only`: the head reads only the per-step `(r_l2, r_l1, contraction)` triples and a small subset of the global `defect_stats` block (via `norm_head`). Tests whether residual *direction* (cosines and projections) matter, or whether scalar norms suffice.
- `single_step`: the iteration is forced to one step and the remaining residual slots are zero-padded. Tests whether unrolling a multi-step fixed-point map matters.
- `untied_residual_blocks`: replaces the shared `T_φ` with `K` untied blocks. Tests whether a true *fixed-point* (parameter-shared) operator is required, or whether `K` generic residual layers suffice.
- `random_update_operator`: freezes `T_φ` at initialisation (no gradients). Tests whether the defects of *any* update operator carry signal, or whether a *learned* operator is required.

## Diagnostics

`forward(x, *, return_path=False)` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit `puzzle_binary` head.
- `prob`: sigmoid of the puzzle logit.
- `defect_features`: shape `(B, stats.full_output_dim)` (or the matching norm-only / final-only shape under the corresponding ablation mode), the readout fed to the head.
- `h_final`: shape `(B, latent_dim)`, the latent at the end of the active iteration.
- `residual_l2`, `residual_l1`: shape `(B, K)`, per-step `||r_k||_2` and mean `|r_k|`.
- `residual_cosine`: shape `(B, K)`, the per-step cosine similarity to the previous defect (entry 0 padded to zero).
- `contraction_ratio`: shape `(B, K)`, the per-step contraction ratio (entry 0 padded to one).
- `residual_signed_delta`: shape `(B, K)`, the per-step signed change in `||r_k||`.
- `residual_projection`: shape `(B, K, projection_dim)`, the per-step residual along the learned projection.
- `path_length`: shape `(B,)`, the total `sum_k ||r_k||_2`.
- `defect_decay`: shape `(B,)`, `||r_0||_2 - ||r_{K-1}||_2`.
- `final_defect_l2`, `final_defect_l1`: shape `(B,)`, the magnitude of the *terminal* defect `r_K = T_φ(h_K, c) - h_K`.
- `oscillation_energy`: shape `(B,)`, the mean of `1 - cos(r_k, r_{k-1})` over the path.
- `defect_stats`: shape `(B, 8)`, the global summary block.
- `active_steps`: shape `(B,)`, integer-valued tensor that records how many iterations actually ran (`1` under `single_step`, `K` otherwise).
- `fixed_point_mode`: shape `(B,)`, integer code identifying the active mode.
- `mechanism_energy`: alias for `final_defect_l2`, the residual-defect energy that operationalises the packet's `logic` mechanism family on the fixed-point readout.
- `proposal_profile_strength`: alias for `path_length`, a single-scalar proxy for residual-trajectory strength.
- `proposal_keyword_count`: scalar preserved for compatibility with the project's research-packet diagnostic schema.

When `return_path=True` the dict additionally contains `h_path`, `r_path`, and `board_embed` so ablation harnesses and probes can inspect the full latent and residual trajectories.

## Contract

- Input: `(B, C, 8, 8)` simple_18 board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Model shapes: latent path `[B, K + 1, latent_dim]`, residual stack `[B, K, latent_dim]`, residual projections `[B, K, projection_dim]`, defect features `[B, stats.full_output_dim]`.
- The puzzle decision flows only through `ψ(x) = stats(h_path, r_path, r_K)` — the head never sees the raw board planes directly except via the encoder's final latent `h_K`, so the residual-defect bottleneck is enforced architecturally.

# Architecture

`Convex Feasibility Residual Network` realises the source packet's unrolled feasibility-projection idea as a bespoke PyTorch model for the repository's `puzzle_binary` task. The classifier never reads raw board planes; it reads the position only through the feasibility geometry of a learned bank of half-space and ball constraints.

## Implementation Binding

- Registered model name: `convex_feasibility_residual_network`
- Source implementation file: `src/chess_nn_playground/models/convex_feasibility.py`
- Idea-local wrapper: `ideas/registry/i094_convex_feasibility_residual_network/model.py`

## Modules

`BoardFeasibilityEncoder` is a compact convolutional trunk over the simple_18 input. It optionally appends two coordinate planes, runs a stack of `Conv2d -> GroupNorm -> GELU` blocks (with optional dropout) at width `channels`, and returns a `latent_dim`-vector via concatenated mean/max pooling and a `Linear -> LayerNorm -> GELU` projection.

`MaterialOnlyEncoder` is the encoder used by the `material_only_encoder` ablation. It replaces the spatial trunk with per-channel sum/mean/max/min summaries that are fed through a two-layer MLP into the same `latent_dim` space.

`LearnedConvexConstraints` parameterises the feasibility bank in latent space:

- `H` half-space normals `n_k in R^{latent_dim}` (row-normalised) with softplus-positive offsets `b_k`.
- `Q` ball constraints expressed as low-rank ambient projections `P_q in R^{m x latent_dim}` (rows row-normalised), centres `mu_q in R^m`, and softplus-positive radii `r_q`.

A parallel set of fixed random constraints (deterministic seed) is stored as buffers and used by the `random_constraints` ablation.

`SoftProjectionLayer` performs the `T`-step unrolled projection. At each step it forms the smoothed hinge violations `v^H_k = h_tau(<n_k, z> - b_k)` and `v^B_q = h_tau(||P_q z - mu_q|| - r_q)`, the sigmoid feasibility gates with sharpness `gamma`, and the gated gradient

`g(z) = sum_k g^H_k v^H_k n_k + sum_q g^B_q v^B_q P_q^T (P_q z - mu_q) / ||P_q z - mu_q||`.

It updates `z_{t+1} = z_t - eta * g(z_t)` and records the path, per-step displacements, and final per-constraint diagnostics. The soft-hinge temperature `tau` enforces `h_tau(u) -> max(0, u)` as `tau -> 0`, so the layer interpolates between a regulariser (large `tau`) and a hard projected-gradient step (small `tau`).

`ConvexFeasibilityResidualNetwork` glues the trunk together: encoder -> projection block -> residual head. The head input is the concatenation `[z ; z_T ; z - z_T ; v ; ||delta||_t ; m]` where `m` is an eight-dimensional projection-geometry summary (residual norm, residual mean-square, total path length, max step norm, mean violation, max violation, violation energy, mean constraint gate). A `LayerNorm -> Linear -> GELU -> Dropout -> Linear -> GELU -> Linear(1)` MLP returns one logit per board. A separate `no_projection` head reads `z` directly when the projection block is bypassed, with comparable parameter count to keep ablations capacity-matched.

## Modes

The `mode` argument selects the active variant:

- `projection` (default): full unrolled feasibility projection with learned half-space and ball banks. The reference implementation called for in the source packet.
- `no_projection`: the head reads `z` directly. Tests whether the projection mechanism adds anything over the encoder.
- `random_constraints`: replaces the learned half-space and ball banks with fixed random constraints (deterministic seed). Tests whether *learned* feasibility regions matter.
- `linear_head_same_params`: the no-projection head with matched parameters. Tests whether capacity alone explains any gain.
- `material_only_encoder`: replaces the convolutional encoder with the material-summary MLP. Tests whether spatial structure is required by the mechanism.

## Diagnostics

`forward(x, *, return_projection=False)` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit `puzzle_binary` head.
- `prob`: sigmoid of the puzzle logit.
- `z`: shape `(B, latent_dim)`, the encoded position before projection.
- `projected_z`: shape `(B, latent_dim)`, the soft-projected latent `z_T`.
- `feasibility_residual`: shape `(B, latent_dim)`, the displacement `z - z_T`.
- `violations`: shape `(B, H + Q)`, the smoothed hinge violations stacked across half-spaces and balls.
- `halfspace_violations`, `ball_violations`: per-bank smoothed hinges.
- `path_step_norms`: shape `(B, T)`, per-step displacement norms.
- `path_length`: shape `(B,)`, total path length over the unrolled projection.
- `residual_norm`, `max_violation`, `mean_violation`, `feasibility_energy`: scalar-per-row geometry summaries.
- `feasible_fraction`, `halfspace_feasible_fraction`, `ball_feasible_fraction`: fraction of constraints satisfied by `z` (raw signed margin <= 0).
- `constraint_gate_mean`: average of the per-constraint feasibility gates.
- `projection_mode`: integer code identifying the active mode (`projection` / `no_projection` / `random_constraints` / `linear_head_same_params` / `material_only_encoder`).
- `mechanism_energy`: `mean(v^2)` — the smoothed feasibility energy that operationalises the packet's `convex` mechanism family.
- `proposal_profile_strength`: residual norm `||z - z_T||`, a single-scalar proxy for the distance-to-feasibility signal.
- `proposal_keyword_count`: integer scalar preserved for compatibility with the project's research-packet diagnostic schema.

When `return_projection=True` the dict additionally contains `projection_path` of shape `(B, T, latent_dim)`, the full residual feature vector, and the per-step constraint gates for ablation harnesses.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Model shapes: encoded latent `[B, latent_dim]`, projection path `[B, T, latent_dim]`, per-constraint violations `[B, H + Q]`.
- The puzzle decision flows only through `psi(z) = [z ; z_T ; z - z_T ; v ; ||delta||_t ; m]` — the head never sees raw board planes, so the feasibility-residual bottleneck is enforced architecturally.

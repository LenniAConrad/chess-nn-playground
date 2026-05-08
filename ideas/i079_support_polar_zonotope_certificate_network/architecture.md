# Architecture

`Support-Polar Zonotope Certificate Network` (SPZC-Net) is a
board-only `puzzle_binary` classifier that follows the markdown
thesis from
`ideas/research_packets/chess_nn_research_2026-04-28_0718_tuesday_new_york_support_polar_zonotope.md`.
It tests the packet's claim that label-2 puzzles can be detected by
whether the board's latent square-pair interaction zonotope protrudes
beyond a learned symmetric polar body, by directly evaluating the
zonotope's closed-form support function against learned directions.

## Mechanism

1. **Board encoder.** `BoardConvStem` over the `simple_18` tensor
   plus a `Linear(channels, d_token)` projection produces 64 square
   tokens `t_i ∈ R^{d_token}`, exactly the role of the `BoardStem` in
   the packet's section-9 PyTorch sketch.
2. **Pair tensor.** For every ordered pair `e = (i, j)` with
   `i != j`, the model concatenates `[t_i, t_j, rho_{ij}]` where
   `rho_{ij} ∈ R^{rel_dim}` is a *fixed relative-square encoding*
   stored as a learned-but-board-independent parameter, giving a
   `(B, 64, 64, 2*d_token + rel_dim)` pair feature tensor.
3. **Generator and gate MLPs.**
   - `phi`: `pair_dim → gen_hidden → d_zono` produces the pair
     generator `g_{ij}(x)`.
   - `gate`: `pair_dim → gate_hidden → 1` followed by `sigmoid`
     produces the activation `a_{ij}(x) ∈ [0, 1]`.
   - The diagonal `i == j` is masked to zero by a fixed `pair_mask`,
     and gated generators are scaled by `1 / sqrt(64 * 63)` so the
     zonotope width remains bounded at initialisation.
4. **Latent center.** A two-layer MLP over the square tokens followed
   by a mean over squares produces `c_x ∈ R^{d_zono}`.
5. **Polar directions and thresholds.** `n_dirs = K` learned
   directions `U ∈ R^{K × d_zono}` are L2-normalised per row, and
   per-direction thresholds `beta_k = softplus(raw_beta_k) + 0.05`
   keep `Q` non-degenerate.
6. **Closed-form support function.** Per-pair projections
   `proj[b, i, j, k] = <u_k, g_{ij}(x)>` and the zonotope center
   projection `cproj[b, k] = <u_k, c_x>` give
   - `width[b, k]   = sum_{i != j} | proj[b, i, j, k] |`,
   - `h_plus[b, k]  = cproj[b, k] + width[b, k] = h_{Z_x}(u_k)`,
   - `h_minus[b, k] = -cproj[b, k] + width[b, k] = h_{Z_x}(-u_k)`,
   - `violations[b] = concat(h_plus - beta, h_minus - beta)` of
     shape `(B, 2K)`.
7. **Residual head.** The puzzle logit is the calibrated monotone
   scalar
   ```
   residual(x) = max_{k, sigma} ( h_{Z_x}(sigma u_k) - beta_k )
   logit(x)    = softplus(raw_scale) * residual(x) + bias.
   ```
   The certificate `(u_k, sigma, beta_k)` and the top contributing
   square pairs by `|proj[b, i, j, k]|` are derived directly from the
   forward pass.
8. **Auxiliary diagnostic head.** A `LayerNorm + MLP` consumes
   `[c_x, cproj, width, h_plus, h_minus, residual, argmax meta]` and
   emits an auxiliary scalar that is *not* added to the puzzle logit;
   it only enriches the prediction artefact.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. Diagnostic
tensors appended to prediction artefacts:

- `residual`: `(B,)` largest learned-direction violation.
- `violations`: `(B, 2K)` concatenation of `h_plus - beta` and
  `h_minus - beta`.
- `h_plus`, `h_minus`: `(B, K)` support-function values
  `h_{Z_x}(u_k)` and `h_{Z_x}(-u_k)`.
- `width`: `(B, K)` zonotope half-width
  `sum_{i != j} |<u_k, g_{ij}>|`.
- `center_projection`: `(B, K)` `<u_k, c_x>`.
- `proj`: `(B, 64, 64, K)` per-pair projections used to extract the
  top contributing square pairs.
- `U`: `(K, d_zono)` row-normalised polar directions.
- `beta`: `(K,)` learned positive thresholds.
- `winning_direction_index`: `(B,)` direction `k` of the argmax
  violation, with `winning_sign ∈ {-1, +1}` and
  `violation_value == residual`.
- `operator_scale`: `(B,)` `softplus(raw_scale)` broadcast across
  the batch (the residual head's monotone slope).
- `auxiliary_logit`: `(B,)` diagnostic-head scalar.
- `gate_mass`: `(B,)` mean activation of the pair gate.
- `ablation_*`: per-batch indicator flags consumed by the packet's
  diagnostic table.

## Ablations

The bespoke builder accepts `model.ablation` in
`{"none", "no_zonotope_width", "single_square_generators",
"random_frozen_directions", "shared_beta", "one_sided",
"no_relative_encoding", "generic_token_baseline",
"certificate_sanity_check"}`, matching the packet's section-11
required ablations. The `generic_token_baseline` and
`certificate_sanity_check` ablations are scaffolded as model-level
indicator flags so the trainer can run matched-budget comparisons
and shuffled-pair stability checks on the same forward pass.

## Implementation Binding

- Registered model name: `support_polar_zonotope_certificate_network`.
- Source implementation file: `src/chess_nn_playground/models/support_polar_zonotope.py`.
- Idea-local wrapper: `ideas/i079_support_polar_zonotope_certificate_network/model.py`.

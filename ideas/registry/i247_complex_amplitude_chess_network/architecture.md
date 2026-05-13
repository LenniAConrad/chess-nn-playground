# Architecture

- Architecture description: i193 dual-stream trunk + CAIO additive primitive
  head, fused through a learned sigmoid gate. The trunk returns the base
  puzzle logit unchanged from i193. The primitive head lifts the board into
  a per-square complex amplitude with chess-rule phase, computes pairwise
  constructive / destructive / curl mass under fixed chess relation masks,
  and runs a small discriminator MLP over the (3R+1)-d fingerprint.

- Input format: simple_18 board tensor `x` of shape `(B, 18, 8, 8)`. CRTK
  metadata is reporting-only and not threaded through the model. The model
  consumes the canonical `batch["x"]` tensor produced by the shared trainer.

- Forward pass:
  1. `trunk_out = ExchangeThenKingDualStream(x)` → `base_logit` and i193
     trunk diagnostics.
  2. `h = CAIOEncoder(x)` (compact GroupNorm + GELU conv stack).
  3. `rho = softplus(mag_proj(h))` and `theta = phase_proj(h) +
     theta_rule(piece colour, side-to-move, square colour)`.
  4. `z = rho * exp(i theta)` per square per amplitude dim.
  5. For each relation mask `M_r`,
     `I_r = Re(outer(z, conj(z)) * exp(i beta_r))` and
     `D_r = Im(...)`, pooled to
     `(constructive_r, destructive_r, curl_r)`.
  6. `z_flipped = encoder(color_flip_simple_18(x))` → `conj_error =
     |z_flipped - z.conj()|`. Used as the 4*3+1 = 13th feature.
  7. `delta = head(fingerprint)` → primitive delta logit per position.
  8. `gate = sigmoid(gate_mlp([trunk_diag, caio_fingerprint]))`,
     initialised near zero.
  9. `logits = base_logit + gate * delta`.

- Tensor shapes:
  - `board`: `(B, 18, 8, 8)`.
  - `h`: `(B, feature_channels, 8, 8)`.
  - `rho, theta`: `(B, amplitude_dim, 8, 8)` and then flattened to
    `(B, amplitude_dim, 64)`.
  - `z`: complex `(B, amplitude_dim, 64)`.
  - per-relation outer product: complex `(B, amplitude_dim, 64, 64)`.
  - `(cons, des, curl)`: each `(B, 4)`.
  - `fingerprint`: `(B, 13)` (4 cons + 4 des + 4 curl + 1 conj_err).
  - `logits`: `(B,)`.

- Output heads: a single puzzle logit `logits` plus the diagnostic dict
  enumerated in `idea.yaml > output_heads`.

- Parameter estimate (default config): i193 trunk ≈ 70k parameters at
  `trunk_channels=64`; CAIOEncoder ≈ 15k at `feature_channels=32, depth=2`;
  `mag_proj` + `phase_proj` 1x1 conv ≈ 2 * 32 * 8 = 512 parameters;
  rule-phase coefficients = 3 * amplitude_dim = 24 parameters; relation
  betas = 4 parameters; head + gate MLPs < 4k parameters. The CAIO
  additive overhead is roughly **+20–25k** parameters on top of i193.

- FLOP estimate: dominated by the relation-masked outer product
  `(B, d, 64, 64)`. For `B=192, d=8` per relation this is roughly
  `1.5e8` real multiplications including the four relations and complex
  arithmetic. Total wall-clock cost ≈ **1.6x i193** (two encoder forwards
  for the conjugacy error + the outer-product pool).

## Implementation Binding

- Registered model name: `complex_amplitude_chess_network`.
- Source implementation: `src/chess_nn_playground/models/primitives/complex_amplitude_chess_network.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`
  (the bespoke i193 `ExchangeThenKingDualStreamNetwork` is wrapped, not
  reimplemented).
- Idea-local wrapper: `ideas/registry/i247_complex_amplitude_chess_network/model.py`.
- Training config: `ideas/registry/i247_complex_amplitude_chess_network/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["complex_amplitude_chess_network"] = build_complex_amplitude_chess_network_from_config`.

- Ablation modes (string `model.ablation` in config.yaml):
  | mode | what it does | falsifies |
  |---|---|---|
  | `none` | Full CAIO primitive | (control) |
  | `real_only` | Force `theta = 0` → real-only bilinear relation head | complex phase is load-bearing |
  | `random_phase` | Replace learned phase with random U(-pi, pi) | chess-rule phase tying matters |
  | `free_phase` | Drop the rule-phase contribution, keep learned phase | rule tying (vs free parameters) matters |
  | `shuffle_relation_masks` | Random permutation of relation mask values | chess relations (not spatial coincidence) drive signal |
  | `no_conjugacy` | Force `conj_error = 0` (skip the colour-flip pass) | conjugacy / Z2 structure matters |
  | `constructive_only` | Zero destructive / curl features | destructive interference is needed |
  | `no_caio` | Zero CAIO fingerprint and gate | primitive adds anything at all |
  | `zero_gate` | Sigmoid gate forced to 0 | gate is doing the work |
  | `trunk_only` | Force `logits = base_logit` (i193 verbatim) | CAIO contributes anything |

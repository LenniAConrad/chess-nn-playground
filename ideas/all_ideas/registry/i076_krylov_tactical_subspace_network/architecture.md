# Architecture

`Krylov Tactical Subspace Network` is a board-only `puzzle_binary`
classifier that asks what happens when a learned chess-structured
linear operator is applied repeatedly to role-conditioned seed
vectors. It follows the markdown thesis from
`ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_2000_saturday_shanghai_krylov_tactical_subspace.md`.

## Mechanism

1. **Board encoder.** A compact convolutional stem (`BoardConvStem`)
   consumes the `simple_18` board tensor and produces a
   `(B, channels, 8, 8)` per-square feature map. The 64 squares give
   `X ∈ R^{B x 64 x channels}`.
2. **Operator builder `A(X)`.** A 64x64 batched operator
   `A = sum_g gate_g(X) * mask_g + U V^T`
   combines five fixed deterministic chess-geometry masks (rook+bishop
   ray, knight, pawn-attack, king, rook-line defense) gated by softplus
   weights from a pooled-board MLP, plus a low-rank context update
   `U V^T` whose factors are linear projections of the per-square
   features. The operator is divided by `max(1, ||A||_F / 8)` to keep
   the spectral norm bounded as the packet recommends. The
   `fixed_operator_only` ablation freezes the gates to the uniform
   distribution and zeros the low-rank update; the
   `random_geometry_operator` ablation swaps the chess masks for a
   fixed random tensor of the same shape.
3. **Role seeds `v_r(X)`.** A linear head reads six role scalars per
   square (`attack`, `defense`, `king_zone`, `high_value_target`,
   `blocker`, `tempo`) and L2-normalises each role vector so the Krylov
   growth curve starts at unit norm.
4. **Krylov block `K_m(A, v_r)`.** A differentiable modified
   Gram-Schmidt Arnoldi block runs for `krylov_steps = m` iterations,
   producing the orthonormal basis `Q_r ∈ R^{B x 64 x m}`, the
   upper-Hessenberg projection `H_r ∈ R^{B x m x m}`, the growth curve
   `||A^k v_r||`, and the final Arnoldi residual. The
   `no_orthogonalization` ablation skips the Gram-Schmidt subtraction
   so the block sees raw powers; the `one_step_only` ablation replaces
   the iteration with a single matrix-vector apply so only first-order
   pressure remains.
5. **Spectral readout.** For each role we expose:
   - **Ritz singular values** of `H_r` (SVD is real, stable, and
     differentiable; the packet explicitly authorises spectra of the
     small projected `H_r` rather than the full 64x64 `A`). The
     `no_spectral_readout` ablation zeros these.
   - **Residual norm** `||A Q_r - Q_r H_r||` (proxy: `h_{m, m-1}`).
   - **Growth curve** `||A^k v_r||`.
   - **Basis energy near the side-to-move king and opposing
     high-value pieces**, computed from `Q_r` and the simple_18 piece
     planes.
6. **Cross-role interaction.** For each cross pair (default:
   `attack/defense`, `attack/king_zone`, `attack/high_value_target`,
   `defense/king_zone`) we report the singular values of the cross-Gram
   matrix `Q_a^T Q_b` (cosines of principal angles) plus the Frobenius
   norm of the cross-Gram. The `no_cross_role_angles` ablation zeros
   this block.
7. **Puzzle head.** A `LayerNorm + MLP` consumes
   `[pool(X), per-role spectral features, cross-role features,
   operator gate weights, operator-norm proxy, low-rank energy]` and
   emits one puzzle logit. The `cnn_same_params` ablation is honoured
   at the trainer level (a size-matched pure-CNN baseline); the bespoke
   model only flips the `ablation_cnn_same_params` flag in the output
   dict.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer. All diagnostic
tensors are finite per batch and are appended to prediction
artefacts:

- `operator_norm`, `operator_low_rank_energy`,
  `operator_gate_weights`: operator diagnostics.
- `role_growth_curves`: `(B, num_roles, m)` `||A^k v_r||`.
- `role_residual_norms`: `(B, num_roles)` Arnoldi residuals.
- `role_ritz_singular_values`: `(B, num_roles, m)` SVs of `H_r`.
- `role_basis_king_energy`, `role_basis_target_energy`:
  `(B, num_roles)` Krylov basis energy on king / high-value-target
  squares.
- `cross_role_principal_angles`: `(B, num_cross_pairs, m)` cosines
  of principal angles `cos(theta) = svd(Q_a^T Q_b)`.
- `cross_role_gram_frobenius`: `(B, num_cross_pairs)` Frobenius norm
  of the cross-Gram matrix.
- `ablation_*`: per-batch indicator flags consumed by the packet's
  diagnostic table.

## Ablations

The bespoke builder accepts `model.ablation in {"none",
"one_step_only", "no_orthogonalization", "fixed_operator_only",
"random_geometry_operator", "no_spectral_readout",
"no_cross_role_angles", "cnn_same_params"}`, matching the packet's
required ablation table. The `cnn_same_params` ablation is enforced
at trainer-level; the model itself only marks the `ablation_*` output
flag.

## Implementation Binding

- Registered model name: `krylov_tactical_subspace_network`.
- Source implementation file: `src/chess_nn_playground/models/krylov_tactical_subspace_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i076_krylov_tactical_subspace_network/model.py`.

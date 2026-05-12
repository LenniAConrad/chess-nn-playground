# Architecture

`Grassmannian Principal-Angle Bottleneck` realises the markdown thesis as
a bespoke model: the central computation is **eigendecomposition** of
role-gated occupied-token covariance matrices followed by **SVD** of
role-pair Gram matrices, yielding principal-angle (canonical-correlation)
spectra between learned subspaces of the Grassmannian `Gr(K, D)`. It is
not a CNN, residual stack, attention block, sheaf, or move-delta
mechanism.

## Pipeline

1. **`Simple18OccupiedTokenExtractor`** decodes the simple_18 board
   tensor into up to `max_tokens = 32` occupied-square tokens with
   deterministic features per token: piece-color one-hots, an own/enemy
   flag, absolute and side-relative coordinates, four castling broadcast
   flags, and an en-passant flag (22 features per token). Padded slots
   carry a zero mask so they never enter downstream pooling or
   covariance.
2. **`PieceSquareTokenEncoder`** maps each token through a 2-layer MLP
   to a `D = token_dim`-dimensional embedding (default 48). The mask is
   re-applied after encoding so embeddings of padded slots are zero.
3. **`RoleGatedCovarianceSubspaces`** computes per-role weighted
   covariance and its top-`K` orthonormal eigenbasis:

   ```
   g_{r,i}(x) = sigmoid(MLP(h_i))_r * mask_i           (B, N, R)
   mu_r       = sum_i g_{r,i} h_i / (sum_i g_{r,i} + eps)
   C_r        = sum_i g_{r,i} (h_i - mu_r)(h_i - mu_r)^T + eps * I_D
   Q_r, lam_r = top-K eigenvectors / eigenvalues of C_r via torch.linalg.eigh
   ```

   Default config has `R = role_count = 8`, `K = subspace_dim = 6`,
   `D = 48`, `eps = covariance_eps = 1e-3`. Eigenvalues are sorted
   descending and the matching eigenvectors are stacked into
   `Q in R^{B x R x D x K}`.
4. **`PrincipalAngleSpectrum`** computes the principal-angle (canonical-
   correlation) spectrum between every unordered role-pair `(a, b)`
   with `a < b`:

   ```
   M_{a,b} = Q_a^T Q_b              in R^{K x K}
   sigma_{a,b} = svdvals(M_{a,b})   clamped to [0, 1] (descending)
   theta_{a,b} = arccos(sigma_{a,b})
   ```

   For `R = 8` roles this gives `P = 28` ordered pairs. Per-pair
   summary statistics (min angle, max angle, mean angle, softmax-cosine
   entropy) are emitted alongside the raw spectra.
5. **`GrassmannianAngleHead`** is a `LayerNorm -> Linear -> ReLU ->
   Linear` MLP over the concatenation of

   - all pair cosine spectra `(P, K)` and pair angle spectra `(P, K)`,
   - per-pair summary statistics `(P, 4)`,
   - per-role eigenvalue spectra `(R, K)` and their logs `(R, K)`,
   - per-role gate masses `(R)`,
   - a 14-dimensional global broadcast vector (side-to-move scalar,
     four castling flags, eight-way en-passant file mask, normalized
     active-token count).

   It returns one puzzle logit; a symmetric `two_class_logits`
   diagnostic is produced by splitting the binary logit so reporting
   can use the binary contract.

## Permutation And Basis-Rotation Invariance

The covariance `C_r` is a sum over tokens, so token reordering does not
change `C_r` (and hence `Q_r`). The principal-angle spectrum
`sigma(Q_a^T Q_b)` is invariant under `Q_r -> Q_r U` for any orthogonal
`U in R^{K x K}` because `(Q_a U)^T (Q_b V) = U^T (Q_a^T Q_b) V` has the
same singular values. Both invariances are verified by the
forward-shape tests for this idea.

## Section 9 Falsifier Ablations

The `ablation` config field selects between the markdown's central
falsifiers. All preserve head input dimensionality so capacity is
matched.

- `no_cross_angles` (markdown's central falsifier): replace pair
  cosine / angle / summary features with zeros while keeping role
  eigenvalues, gate masses, and the global broadcast. If this matches
  the main model, the cross-subspace geometry is not what carries the
  signal.
- `eigenvalues_only`: identical signal-removal to `no_cross_angles`;
  retained as a separately named ablation so report tooling that
  references either falsifier finds a working config.
- `batch_shuffled_angles`: shuffle the per-sample principal-angle
  feature block across the batch, so angle structure is no longer tied
  to the sample's roles.
- `pooled_token_head`: bypass the subspace machinery and feed
  mean / max / std token-pooled embeddings through a learned linear
  projection to the same head input dimensionality.
- `no_orthonormalization`: replace `Q_a^T Q_b` with the rank-1 outer
  product of the unit-normalized role means, so the principal-angle
  SVD reduces to a single cosine and basis-rotation invariance is
  destroyed.

## Output Contract

`forward(x)` returns a dictionary including `logits` of shape `(B,)`
for `num_classes=1`, `two_class_logits`, the raw
`principal_angle_cosines` and `principal_angle_radians` `(B, P, K)`,
per-pair `pair_min_angle / pair_max_angle / pair_mean_angle /
pair_entropy` `(B, P)`, per-role `role_eigenvalues` `(B, R, K)` and
`role_gate_mass` `(B, R)`, the `active_token_count`,
`mean_pair_cosine`, `mean_pair_angle`, `pair_mean_angle_std`,
`mechanism_energy`, `eigen_mass`, and `ablation_active`. Engine,
verification, source, and CRTK metadata are never used as input.

## Why This Is Not A Generic CNN Variant

The central operator is **batched eigendecomposition + SVD of
role-pair Gram matrices**, not convolution, residual stacking, square
attention, or move enumeration. The model is permutation-invariant
over occupied tokens and basis-rotation invariant inside each role
subspace, both of which are properties of Grassmannian geometry rather
than CNN feature pooling. The `no_orthonormalization` ablation isolates
exactly the orthonormalization step, and `no_cross_angles`
ablation removes the cross-subspace geometry while keeping every other
diagnostic, so any positive result attributable to Grassmannian
geometry must beat both falsifiers.

## Implementation Binding

- Registered model name: `grassmannian_principal_angle_bottleneck`
- Source implementation: `src/chess_nn_playground/models/grassmannian_principal_angle_bottleneck.py`
- Idea-local wrapper: `ideas/registry/i061_grassmannian_principal_angle_bottleneck/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_grassmannian_principal_angle_bottleneck_from_config`.
- The idea-local wrapper does not import or call the shared
  `ResearchPacketProbe` / `build_research_packet_probe_from_config` scaffold.

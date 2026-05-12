# Architecture

`Polar-Procrustes Alignment Bottleneck` realises the markdown thesis as
a bespoke model: the central computation is **batched SVD of the
cross-covariance ``C(x) = X(x)^T Y(x)`` of learned own / opponent role
matrices** built from current-board occupied tokens, recovering the
optimal orthogonal Procrustes alignment ``Q* = U V^T`` and the polar
strain ``H = V Sigma V^T``. It is not a CNN, residual stack, attention
block, sheaf, transport, matrix-pencil, generalized-spectrum,
principal-angle, or move-delta mechanism.

## Pipeline

1. **`Simple18OwnOpponentTokenExtractor`** decodes the simple_18 board
   tensor into up to ``max_tokens = 32`` occupied-square tokens with
   deterministic features (12 piece-color one-hots, an own/enemy flag,
   absolute and side-relative coordinates, four castling broadcast
   flags, and an en-passant flag — 22 features per token). It also
   emits per-token side-to-move-relative ``own_mask`` and ``opp_mask``
   so the role pooler never mixes sides. Padded slots carry zero
   masks.
2. **`PieceSquareTokenEncoder`** maps each token through a 2-layer MLP
   to a ``D = token_dim``-dimensional embedding (default 48). The mask
   is re-applied after encoding.
3. **`RoleMatrixPooler`** builds the own / opponent role matrices via
   masked-softmax learned role queries:

   ```
   own_logits(h) = own_query(h)              (B, N, R)
   opp_logits(h) = opp_query(h)              (B, N, R)
   own_weights[n, r] = softmax_n(own_logits + side_mask_bias)[n, r]
   opp_weights[n, r] = softmax_n(opp_logits + side_mask_bias)[n, r]
   X[r, :] = sum_n own_weights[n, r] * h_n
   Y[r, :] = sum_n opp_weights[n, r] * h_n
   ```

   Wrong-side and padded tokens are zeroed out by a large negative
   bias before the softmax, so each role pulls only from its side. The
   sum is permutation-invariant in token order. Role mass per side
   feeds the head as a separate signal.
4. Optionally row-normalise ``X`` and ``Y`` (LayerNorm-style mean /
   unit-variance per role row) when ``normalize_rows`` is true so the
   cross-covariance is not dominated by row scale.
5. **`PolarProcrustesLayer`** computes the cross-covariance and solves
   the orthogonal Procrustes problem via SVD:

   ```
   C = X^T Y / R       (matrix_space = "embedding", shape (B, D, D))
   C = X Y^T / D       (matrix_space = "role",      shape (B, R, R))
   U, Sigma, V^T = svd(C + cross_cov_eps * diag(1, 2, ..., M) / M)
   Q* = U V^T
   H  = V Sigma V^T
   ```

   The diagonal tilt is below the typical signal scale and breaks any
   coincident singular values so ``torch.linalg.svd`` backward stays
   finite. The layer also returns the Procrustes residual
   ``||X Q* - Y||_F`` (or ``||Q* X - Y||_F`` in role-space), the
   identity residual ``||X - Y||_F``, the alignment improvement
   ``identity - procrustes``, per-role residuals, the singular values
   and polar diagonal of ``H``, the nuclear / spectral / stable-rank
   summaries of ``C``, the Frobenius norms ``||X||_F`` / ``||Y||_F``,
   and the separate singular values of ``X`` and ``Y``.
6. **`PolarProcrustesHead`** is a ``LayerNorm -> Linear -> ReLU ->
   Linear`` MLP over the concatenation of

   - the procrustes block: per-role residual ``(R)`` + scalar summary
     ``[procrustes, identity, improvement, ||X||_F, ||Y||_F, nuclear,
     spectral, stable_rank]`` ``(8)``,
   - the spectrum block: singular values of ``C`` ``(M)`` + diagonal
     of ``H`` ``(M)``,
   - the separate-stats block: singular values of ``X`` ``(R)`` + of
     ``Y`` ``(R)`` + own / opp role mass ``(R)`` + ``(R)`` when
     ``include_separate_spectra`` is true,
   - a 15-dim global broadcast vector (side-to-move scalar, four
     castling flags, eight-way en-passant file mask, normalised
     active-token count, normalised own-token fraction, normalised
     opp-token fraction).

   It returns one puzzle logit; a symmetric ``two_class_logits``
   diagnostic is produced by splitting the binary logit so reporting
   can use the binary contract.

## Permutation Invariance And Alignment Properties

Role pooling computes ``X = sum_n softmax_n(...) * h_n`` over occupied
tokens (per side), so reordering active tokens within a sample does not
change ``X`` or ``Y``. The cross-covariance and SVD are sample-local.
The Procrustes residual ``||XQ - Y||_F`` and the singular values of
``C = X^T Y`` are invariant to a shared right-orthogonal coordinate
change ``X -> X W``, ``Y -> Y W``: the feasible set of orthogonal ``Q``
is closed under conjugation by ``W`` and the Frobenius norm is
orthogonally invariant, so the alignment quantifies relative geometry
rather than embedding axes.

## Section 9 Falsifier Ablations

The ``ablation`` config field selects between the markdown's central
falsifiers. All preserve head input dimensionality so capacity is
matched.

- ``separate_matrix_stats_only`` (markdown's central falsifier): zero
  the cross-covariance / Procrustes / spectrum blocks while keeping
  the separate singular values, role mass and global broadcast.
- ``identity_alignment_only``: replace ``Q*`` with the identity, so
  only ``||X - Y||_F`` and identity-aligned per-role residuals enter
  the head.
- ``random_orthogonal_alignment``: replace ``Q*`` with a deterministic
  batch-shared random orthogonal matrix (built once via QR of a fixed
  Gaussian seed and registered as a buffer).
- ``batch_shuffled_opponent``: pair each ``X(x)`` with ``Y(x')`` from a
  deterministic batch permutation and rebuild the spectrum so the
  matrix pair no longer comes from the same sample.
- ``material_only_matrices``: zero the coordinate / castling /
  en-passant / own-flag features in each token before encoding so only
  piece-color identity remains.
- ``role_pool_mean_only``: replace learned role queries with a
  deterministic projection of mean / max / std side-wise pools to the
  same role-matrix shape.
- ``singular_values_only``: keep only the singular values and polar
  diagonal of ``C``; zero the residual / improvement / Q*-derived
  block and the separate-stats block.

## Output Contract

``forward(x)`` returns a dictionary including ``logits`` of shape
``(B,)`` for ``num_classes=1``, ``two_class_logits``, ``cross_covariance``
``(B, M, M)``, ``orthogonal_alignment`` ``(B, M, M)``,
``singular_values`` ``(B, M)``, ``polar_strain_diagonal`` ``(B, M)``,
``procrustes_residual``, ``identity_residual``,
``alignment_improvement``, ``per_role_residual`` ``(B, R)``,
``nuclear_norm``, ``spectral_norm``, ``stable_rank``, ``x_norm``,
``y_norm``, ``x_singular_values`` ``(B, R)``, ``y_singular_values``
``(B, R)``, ``own_role_mass`` ``(B, R)``, ``opp_role_mass`` ``(B, R)``,
``active_token_count``, ``mechanism_energy`` and ``ablation_active``.
Engine, verification, source, and CRTK metadata are never used as
input.

## Why This Is Not A Generic CNN Variant

The central operator is **batched SVD of the cross-covariance
``C = X^T Y``** with explicit polar / Procrustes recovery (``Q* = U V^T``,
``H = V Sigma V^T``), not convolution, residual stacking, square
attention, sheaf propagation, transport, matrix pencils,
generalized-eigenvalue problems, principal-angle SVD between
orthonormal subspaces, or move enumeration. The
``separate_matrix_stats_only`` ablation removes exactly the
cross-covariance signal while keeping each side's standalone
spectrum, the ``identity_alignment_only`` and
``random_orthogonal_alignment`` ablations remove sample-specific
optimal orthogonal alignment, and ``batch_shuffled_opponent`` breaks
sample-specific own / opp pairing — any positive result attributable
to polar-Procrustes alignment must beat all four falsifiers.

## Implementation Binding

- Registered model name: `polar_procrustes_alignment_bottleneck`
- Source implementation: `src/chess_nn_playground/models/trunk/polar_procrustes_alignment_bottleneck.py`
- Idea-local wrapper: `ideas/registry/i063_polar_procrustes_alignment_bottleneck/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_polar_procrustes_alignment_bottleneck_from_config`.
- The idea-local wrapper does not import or call the shared
  `ResearchPacketProbe` / `build_research_packet_probe_from_config` scaffold.

# Architecture

`Matrix-Pencil Generalized Spectrum Bottleneck` realises the markdown
thesis as a bespoke model: the central computation is **batched
Cholesky + symmetric eigendecomposition of the whitened pencil matrix**
``L^{-1} A(x) L^{-T}`` for a learned PSD matrix pair ``(A(x), B(x))``
built from current-board occupied tokens, plus generalized Rayleigh
quotients ``z^T A z / z^T B z`` along learned probe directions. It is
not a CNN, residual stack, attention block, sheaf, or move-delta
mechanism.

## Pipeline

1. **`Simple18OccupiedTokenExtractor`** decodes the simple_18 board
   tensor into up to ``max_tokens = 32`` occupied-square tokens with
   deterministic features per token: piece-color one-hots, an own/enemy
   flag, absolute and side-relative coordinates, four castling broadcast
   flags, and an en-passant flag (22 features per token). Padded slots
   carry a zero mask so they never enter downstream pooling or matrix
   construction.
2. **`PieceSquareTokenEncoder`** maps each token through a 2-layer MLP
   to a ``D = token_dim``-dimensional embedding (default 64). The mask
   is re-applied after encoding so embeddings of padded slots are zero.
3. **`LowRankBoardMatrixPair`** builds the learned PSD matrix pair from
   masked-softmax token summaries:

   ```
   weight_logits_r(h) = linear(h)                  (B, N, K)
   value_r(h)         = linear(h)                  (B, N, M)
   weight_r           = softmax over occupied tokens of weight_logits_r
   U_r[k, :]          = sum_i weight_r[i, k] * value_r[i, :]
   A(x)               = U_A^T U_A / K + eps * I_M
   B(x)               = U_B^T U_B / K + eps * I_M
   ```

   Default config has ``K = factor_rank = 16``, ``M = matrix_dim = 16``,
   ``D = 64``, ``eps = matrix_eps = 1e-3``. Both ``A`` and ``B`` are
   symmetrised and isotropically regularised so ``B`` is positive
   definite by construction.
4. **`GeneralizedSpectrumLayer`** computes the generalized eigenvalues
   of the pencil ``(A, B)`` via the whitened symmetric form:

   ```
   L     = cholesky(B)
   Y     = solve_triangular(L, A, upper=False)
   C     = solve_triangular(L, Y^T, upper=False)^T
   C_sym = 0.5 * (C + C^T)
   lambda(x) = eigvalsh(C_sym)            (B, M, descending after flip)
   ```

   The layer also returns separate eigenvalue spectra of ``A`` and
   ``B`` (descending) and, when ``include_rayleigh_probes`` is true,
   generalized Rayleigh quotients

   ```
   R_p(x) = z_p^T A(x) z_p / z_p^T B(x) z_p   (B, P)
   ```

   along ``probe_count`` learned probe directions ``z_p`` normalized to
   unit length.
5. **`MatrixPencilHead`** is a ``LayerNorm -> Linear -> ReLU -> Linear``
   MLP over the concatenation of

   - generalized eigenvalues and log generalized eigenvalues ``(M)`` each,
   - pencil summary statistics: spread ``max - min``, condition-like
     ratio ``max / min``, ``tr(A) / tr(B)``, ``||A||_F``, ``||B||_F``
     (5 dims),
   - separate eigenvalues of ``A`` and ``B`` ``(M)`` each plus their
     diagonals ``(M)`` each (4 * M total) when
     ``include_separate_spectra`` is true,
   - generalized Rayleigh probe quotients ``(P)``,
   - a 14-dimensional global broadcast vector (side-to-move scalar,
     four castling flags, eight-way en-passant file mask, normalized
     active-token count).

   It returns one puzzle logit; a symmetric ``two_class_logits``
   diagnostic is produced by splitting the binary logit so reporting
   can use the binary contract.

## Permutation Invariance And Spectrum Properties

The factor builders compute ``U_r = sum_i softmax_i(logits) * value_i``
over occupied tokens, so reordering the active token slots does not
change ``U_r`` (and hence ``A``, ``B``, or the generalized spectrum).
Because ``B`` is regularised to be positive definite, the generalized
eigenvalues of ``(A, B)`` are real, equal the stationary values of the
generalized Rayleigh quotient ``R(v) = (v^T A v) / (v^T B v)``, and are
invariant under congruent changes of coordinates applied to both forms.

## Section 9 Falsifier Ablations

The ``ablation`` config field selects between the markdown's central
falsifiers. All preserve head input dimensionality so capacity is
matched.

- ``separate_spectra_only`` (markdown's central falsifier): zero the
  generalized eigenvalues, log generalized eigenvalues, pencil summary
  statistics, and Rayleigh probe quotients while keeping the separate
  eigenvalues / diagonals of ``A`` and ``B``. If this matches the main
  model the relative pencil geometry is not what carries the signal.
- ``trace_ratio_only``: zero everything except ``spread``, ``cond``,
  ``tr(A)/tr(B)``, ``||A||_F`` and ``||B||_F``, so the head sees only
  scalar size summaries.
- ``batch_shuffled_b``: pair each ``A(x)`` with a ``B(x')`` from a
  deterministic batch permutation and rebuild the spectrum so the
  matrix pair no longer comes from the same sample.
- ``random_factors``: freeze the token encoder and matrix-pair builders
  at initialization (``requires_grad=False``) so only the head trains.
- ``single_matrix_spectrum``: keep only the eigenvalues, diagonal and
  Frobenius norm of ``A`` (no ``B``, no pencil features).
- ``mean_pool_head``: bypass the pencil and feed mean / max / std
  token-pooled embeddings through a learned linear projection of the
  same dimensionality as the pencil features.
- ``material_only_tokens``: zero the coordinate / castling / en-passant
  / own-flag features in each token before encoding, so only piece
  identity remains.

## Output Contract

``forward(x)`` returns a dictionary including ``logits`` of shape
``(B,)`` for ``num_classes=1``, ``two_class_logits``, the raw
``generalized_eigenvalues`` and ``log_generalized_eigenvalues``
``(B, M)``, ``rayleigh_probes`` ``(B, P)``, ``matrix_a`` and
``matrix_b`` ``(B, M, M)``, ``eigenvalues_a`` and ``eigenvalues_b``
``(B, M)``, ``trace_a``, ``trace_b``, ``trace_ratio``, ``condition_b``,
``proportionality_diagnostic`` ``(B,)``, ``mechanism_energy``,
``active_token_count``, and ``ablation_active``. Engine, verification,
source, and CRTK metadata are never used as input.

## Why This Is Not A Generic CNN Variant

The central operator is **batched Cholesky + symmetric
eigendecomposition of the whitened pencil ``L^{-1} A L^{-T}``**, not
convolution, residual stacking, square attention, sheaf propagation, or
move enumeration. The ``separate_spectra_only`` ablation removes
exactly the generalized eigenvalues / Rayleigh quotients while keeping
each matrix's standalone spectrum, the ``batch_shuffled_b`` ablation
breaks sample-specific pairing, and ``random_factors`` removes learned
matrix construction; any positive result attributable to matrix-pencil
geometry must beat all three falsifiers.

## Implementation Binding

- Registered model name: `matrix_pencil_generalized_spectrum_bottleneck`
- Source implementation: `src/chess_nn_playground/models/matrix_pencil_generalized_spectrum_bottleneck.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i062_matrix_pencil_generalized_spectrum_bottleneck/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_matrix_pencil_generalized_spectrum_bottleneck_from_config`.
- The idea-local wrapper does not import or call the shared
  `ResearchPacketProbe` / `build_research_packet_probe_from_config` scaffold.

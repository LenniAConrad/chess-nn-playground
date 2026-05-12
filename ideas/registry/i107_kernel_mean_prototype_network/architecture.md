# Architecture

`Kernel Mean Prototype Network` (KMPN) treats a chess position as the
empirical distribution of its occupied piece tokens and embeds that set as a
single kernel mean in a learned reproducing-kernel feature space.  No piece
attends to another piece, no pairwise transport plan is computed, and no
spatial convolution scans the board: the entire bottleneck is the comparison
between the kernel mean and a small bank of learnable prototype embeddings.

## Tensor Contract

```text
input:                     (B, 18, 8, 8)
square tokens:             (B, 64, token_dim)
kernel features phi(x_i):  (B, 64, phi_dim)
occupancy mask:            (B, 64)
kernel mean mu(x):         (B, phi_dim)
prototype distances d_p:   (B, num_prototypes)
prototype similarities s_p:(B, num_prototypes)
diagnostic vector:         (B, 1 + 6 + 1 + 1 + 2 * num_prototypes)
logits:                    (B,)
```

`token_dim`, `phi_dim`, and `num_prototypes` are configurable.  The model is
strictly board-only and never reads engine, search, source, or CRTK metadata.

## Components

- Square tokenizer: each of the 64 squares concatenates its 18 board planes
  with six deterministic geometric coordinates (rank, file, centred rank,
  centred file, edge distance, square colour) and is passed through a
  two-layer MLP with `LayerNorm` and `GELU` to produce a token of dimension
  `token_dim`.  Empty squares are encoded too but are masked out before the
  kernel mean.
- Kernel feature lift `phi`: a learnable random-Fourier-style map
  ``phi(t) = sqrt(2 / m) * cos(W t + b)`` where ``W`` is initialised from a
  Gaussian (scaled by ``1 / bandwidth``) and ``b`` is uniform on
  ``[0, 2 * pi)``.  Both ``W`` and ``b`` remain trainable so the kernel can
  adapt to puzzle vs non-puzzle structure while keeping the random-feature
  ``cos`` non-linearity that approximates an RBF inner product.
- Occupancy: the union of the 12 piece planes is flattened to a (B, 64) mask
  that selects which kernel features participate in the empirical mean.  The
  cardinality ``N(x)`` of the occupied set is recorded as a diagnostic.
- Kernel mean embedding: ``mu(x) = (1 / max(N(x), eps)) * sum_i m_i phi(x_i)``
  is a single (B, phi_dim) vector.  This is the only signal the classifier
  sees about the piece set; replacing pieces, swapping squares, or padding
  with empties without changing the empirical kernel mean cannot change the
  forward pass.
- Prototype bank: ``P`` learnable prototype vectors ``mu_p`` live in the same
  R^{phi_dim} as the kernel mean.  Squared MMD-like distances
  ``d_p = ||mu(x) - mu_p||^2`` and per-prototype RBF similarities
  ``s_p = exp(-gamma_p * d_p)`` are computed with one trainable bandwidth per
  prototype (parameterised through `log_gamma` so the bandwidths stay
  positive).
- Set diagnostics: log-cardinality of the occupied set, six side-canonical
  per-piece-type counts (us minus them for K, Q, R, B, N, P) which
  preserve the side-to-move contract, the scalar us-vs-them imbalance,
  the kernel self-similarity ``||mu(x)||^2``, and the full
  prototype-distance / similarity vectors.
- Classifier: `LayerNorm` + `Linear(head_hidden)` + `GELU` + dropout +
  `Linear(1)` over the concatenation of the kernel mean ``mu(x)`` and the
  diagnostic vector.  Returns one puzzle logit per board.

## Output Diagnostics

Forward returns `logits` plus `kernel_mean`, `kernel_features`,
`occupancy_mask`, `occupied_count`, `log_occupied_count`,
`canonical_piece_counts`, `us_them_imbalance`, `kernel_self_similarity`,
`prototype_distances`, `prototype_similarities`, `prototype_log_gamma`, and
`diagnostic_features`.

## Implementation Binding

- Registered model name: `kernel_mean_prototype_network`.
- Source implementation file: `src/chess_nn_playground/models/kernel_mean_prototype_network.py`.
- Idea-local wrapper: `ideas/registry/i107_kernel_mean_prototype_network/model.py`.

# Architecture

`Zobrist Kernel Feature Network` consumes the simple_18 board tensor, builds
``M`` fixed Zobrist-style random feature banks over the 12 piece-square
planes, lifts each per-bank fingerprint through a fixed random projection plus
a sin/cos nonlinearity (random Fourier features), and trains only a small
classifier MLP over the concatenated kernel features.  The Zobrist code
banks, the projection matrices, and the phase biases are non-trainable
buffers — only the classifier head is learned.

## Input

- Repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- Only the 12 piece planes
  `(P, N, B, R, Q, K, p, n, b, r, q, k)` are consumed; the remaining 6
  metadata planes (side-to-move, castling rights, en passant, etc.) are
  ignored to honour the classical piece-square occupancy semantics of Zobrist
  hashing.
- CRTK / source / engine metadata is ignored — only the board tensor is
  consumed by the model.

## Fixed Zobrist Code Banks

For each bank `m = 1..M` the constructor samples a Rademacher tensor
`Z_m in {-1, +1}^{12 x 64 x D}` from a fixed seeded generator and stores it as
a non-trainable buffer.  `D` is the per-bank `feature_dim`.  At forward time
the per-bank fingerprint of an input board with flattened piece occupancy
`O \in {0, 1}^{12 x 64}` is

```
s_m = sum_{p, s} O[p, s] * Z_m[p, s] \in R^D.
```

This is the differentiable, addition-based analogue of XOR-ing the Zobrist
codes of every occupied piece-square pair.  Per the math thesis,
`<s_m, s_m'>` is, in expectation, `D * |occ(B) cap occ(B')|`, so `s_m` is a
random projection of the piece-square occupancy intersection kernel.

## Random Fourier Kernel Features

Each bank also stores a fixed random projection matrix
`W_m ~ N(0, 1/D) \in R^{D x D}` and a fixed phase bias
`b_m ~ U[0, 2\pi) \in R^D`.  The bank's kernel feature is

```
arg_m       = bandwidth * (W_m s_m) + b_m
phi_m       = (1 / sqrt(D)) * [cos(arg_m), sin(arg_m)]   \in R^{2D}.
```

This is the standard Rahimi-Recht random Fourier feature approximation of an
RBF kernel evaluated on the Zobrist fingerprint.  `bandwidth` controls the
inverse RBF bandwidth.

## Diagnostic Side Channel And Classifier

The fusion vector concatenates

- `phi_1, ..., phi_M` — the `M * 2D` random Fourier features;
- per-bank raw fingerprint norms `||s_m||_2` (length `M`);
- per-bank kernel feature norms `||phi_m||_2` (length `M`);
- the global occupancy count `sum_{p, s} O[p, s]` (length `1`).

A two-layer GELU MLP with optional `BatchNorm1d`, dropout, and a final
`Linear` projection reads the fusion vector and emits one `puzzle_binary`
logit.  All Zobrist codes, projection matrices, and phase biases are buffers
(`persistent=True`); only the classifier MLP weights are trainable.

The forward pass returns a dict whose `logits` tensor has shape `(batch,)`
alongside diagnostics including `occupancy_count`, `fingerprint_total_norm`,
`kernel_feature_total_norm`, `cos_feature_mean`, `sin_feature_mean`,
`fingerprint_mean_abs`, `per_bank_signature_norm_mean`,
`per_bank_kernel_norm_mean`, and `signature_norm_bank_<m>` /
`kernel_norm_bank_<m>` for each bank, all of shape `(batch,)`, for ablation
analysis.

## Implementation Binding

- Registered model name: `zobrist_kernel_feature_network`.
- Source implementation: `src/chess_nn_playground/models/zobrist_kernel_feature_network.py`.
- Idea-local wrapper: `ideas/registry/i135_zobrist_kernel_feature_network/model.py`.

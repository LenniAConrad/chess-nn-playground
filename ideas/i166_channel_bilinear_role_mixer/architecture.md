# Architecture

`Channel-Bilinear Role Mixer` is a board-only classifier for the
`puzzle_binary` task. It accepts the repo's simple 18-plane current-board
tensor with shape `(B, 18, 8, 8)` and returns one puzzle logit per position.

The trunk is a compact CNN: a `Conv3x3 -> BatchNorm -> GELU` stem followed by
`depth` `ConvBlock`s (`Conv3x3 -> BatchNorm -> GELU -> Dropout`) of width
`channels`. The trunk outputs per-square channel features
`H \in R^{C \times 8 \times 8}`.

`K = num_roles` learned roles each consist of

- a softmax spatial gate `m_k(s)` over the 64 squares, and
- a per-role channel projection `W_k \in R^{D \times C}` with bias
  `b_k \in R^D`.

The role summary is the LayerNorm of a softly pooled, projected channel vector

```text
r_k = LayerNorm( W_k * sum_{s} m_k(s) H(s) + b_k )      (1)
```

so each role is a learned ``role-aware spatial average over channels``. The
softmax gate keeps `r_k` a true weighted mean of channel features rather than
an unbounded sum; the projection lets every role see a different linear
combination of the trunk channels.

The bilinear head explicitly models pairwise role interactions. Two shared
projections `U, V \in R^{R \times D}` produce two rank-`R` views of every
role,

```text
P_k = U r_k,          Q_k = V r_k                       (2)
```

and the asymmetric pairwise interaction matrix is

```text
M_{ij} = (1 / sqrt(R)) * <P_i, Q_j>                      (3)
```

so `M \in R^{K \times K}` is exactly the table of all ordered role-pair
interactions, expressed via the low-rank factorisation
`W_{ij} = U^T \mathrm{diag}(\cdot) V`. Two distinct projections `U, V` keep
the interaction asymmetric: the ``own-rooks attacking opponent-king-zone``
direction is not forced to equal its converse.

The flattened `K^2` interaction scores are passed to a small MLP head:
`LayerNorm -> Linear(K^2, hidden_dim) -> GELU -> Dropout -> Linear` to one
puzzle logit. No square-pair tensor and no local product convolution is ever
materialised; pairwise interactions are computed only between the `K` role
summaries.

Parameters scale as `O(K * D * R + K^2)` for the head, which is materially
cheaper than a dense `K`-by-`K` bilinear form `O(K^2 D^2)` while still
representing every ordered role-pair interaction.

## Diagnostics

The forward pass returns a dict with the following keys (`B = batch`):

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` for `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `trunk_features`: `(B, channels, 8, 8)` post-trunk feature map.
- `role_gates`: `(K, 8, 8)` softmax spatial gates `m_k(s)`.
- `role_pooled_channels`: `(B, K, channels)` per-role spatial pool before
  projection.
- `role_summaries_pre_norm`: `(B, K, role_dim)` projected role vectors before
  LayerNorm.
- `role_summaries`: `(B, K, role_dim)` final role summaries.
- `bilinear_left`: `(B, K, bilinear_rank)` left view `P_k`.
- `bilinear_right`: `(B, K, bilinear_rank)` right view `Q_k`.
- `bilinear_interaction_matrix`: `(B, K, K)` matrix `M_{ij}`.
- `bilinear_diag`: `(B, K)` self-interactions `M_{kk}`.
- `bilinear_energy`: `(B,)` mean of `M_{ij}^2`.
- `bilinear_off_diag_energy`: `(B,)` mean of squared off-diagonal entries.
- `bilinear_asymmetry`: `(B,)` mean of `(M - M^T)^2`.
- `role_magnitude`: `(B, K)` `||r_k||`.
- `role_gate_entropy`: `(B,)` mean entropy of the softmax role gates.
- `depth_levels`: `(B,)` scalar tag of the configured depth.

## Implementation Binding

- Registered model name: `channel_bilinear_role_mixer`
- Source implementation file: `src/chess_nn_playground/models/channel_bilinear_role_mixer.py`
- Idea-local wrapper: `ideas/i166_channel_bilinear_role_mixer/model.py`

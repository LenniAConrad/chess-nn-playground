# Math Thesis

Channel-Bilinear Role Mixer

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `5`.

## Working thesis

Ordinary classifier heads pool channels additively: the global pool over a
trunk feature map ``H \in R^{C \times 8 \times 8}`` collapses to a vector
``\bar h \in R^C`` and the head is a linear map of that vector. Pairwise
interactions between distinct ``role channels`` -- such as own heavy-piece
features against opponent king-zone features -- can only appear after
non-linearities mix them. A low-rank bilinear head can model those pairwise
role interactions explicitly, *without* materialising any
``square-pair tensor`` of shape ``(64, 64)`` or any local product
convolution.

## Construction

Let ``K = num_roles``. Each role ``k = 1, ..., K`` is parametrised by

- a softmax spatial gate ``m_k(s) = softmax_s(g_k(s))`` over the 64 squares,
- a per-role channel projection ``W_k \in R^{D \times C}`` with bias
  ``b_k \in R^D``.

The role summary is

```text
r_k = LayerNorm( W_k * sum_{s=1..64} m_k(s) H(s) + b_k ) \in R^D       (1)
```

Two shared rank-``R`` projections ``U, V \in R^{R \times D}`` give two views
of every role,

```text
P_k = U r_k,        Q_k = V r_k \in R^R                                  (2)
```

and the asymmetric pairwise interaction matrix is

```text
M_{ij} = (1 / sqrt(R)) * <P_i, Q_j>                                      (3)
```

which is the explicit low-rank bilinear form

```text
M_{ij} = (1 / sqrt(R)) * r_i^T (U^T V) r_j                               (4)
```

i.e. every ordered pair ``(i, j)`` has its own scalar interaction expressed
via the shared low-rank factor ``U^T V``. The scaling ``1 / sqrt(R)`` keeps
``M_{ij}`` at unit scale at initialisation. Asymmetry between the two
directions ``i \to j`` and ``j \to i`` is preserved because ``U`` and ``V``
are independent: the head can encode that
``own-rooks targeting opponent-king-zone`` is different from its converse.

The full ``K \times K`` matrix ``M`` is flattened and passed to a small MLP
head, producing one logit. No quadratic-in-squares tensor is constructed:
all pairwise interactions are computed between the ``K`` role summaries and
the head cost is ``O(K * D * R + K^2)``.

## Why this matters

Plain CNN heads cannot represent ``does role i co-activate with role j``
without committing capacity to non-linear feature mixing in the trunk. The
bilinear head expresses that question directly with a low-rank parameter
budget, while the trunk and role pooling stay close to a standard CNN. Roles
are *learned* (via the gates and projections), so this is not a hand-coded
piece-class dictionary -- but the architecture exposes the role pool, the
left/right views, and the interaction matrix as diagnostics so the learned
role behaviour can be inspected directly.

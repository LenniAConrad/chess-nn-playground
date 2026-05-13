# Math Thesis

Source: `ideas/research/primitives/external_32_elementary_symmetric_gibbs_hodge_primitives.md`,
rank-2 proposal `primitive_gibbs_cut_partition`. The rank-1 proposal in
the same packet is the same elementary-symmetric polynomial operator as
`p024 event_symmetric_interaction_accumulator`, so it is not promoted
here.

## Working thesis

For a fixed grid `G = (V, E)` with `H` rows and `W` columns, edge costs
`c in R_+^{B x E x d_cut}`, source penalties `s in R^{B x V x d_cut}`,
and sink penalties `t in R^{B x V x d_cut}`:

    Z_{b,h} = sum_{S subseteq V} exp( - ( sum_{(i,j) in cut(S)} c_{b,ij,h}
                                          + sum_{i not in S} s_{b,i,h}
                                          + sum_{i in S}     t_{b,i,h} ) / tau ),
    y_{b,h} = -tau * log Z_{b,h},
    m_{b,e,h} = dy_{b,h} / dc_{b,e,h}    (Gibbs cut-edge probability).

The marginals `m_e` are exactly the Gibbs probability that edge `e`
crosses the random cut sampled from the distribution
`p(S) propto exp(-energy(S)/tau)`. We do not materialise them as a
separate tensor; autograd through `y` yields them directly when needed.

## Reduction via row transfer DP

Direct evaluation requires `|V| = H * W` bits per subset, so `2^(H*W)`
configurations. The row transfer matrix reduces this to `O(H * 2^(2W))`
by treating each row's column-membership as a state in `{0, ..., 2^W-1}`.

For the bit decomposition `bits[S, j] = (S >> j) & 1`:

- *Within-row cost* (`(S, j)` -> horizontal edge `(r, j) - (r, j+1)`):

  ```
  within_cost[S]   = sum_{j<W-1}  c_h[r, j] * |bits[S, j] - bits[S, j+1]|
  ```

- *Cell cost* (per-vertex source / sink penalty):

  ```
  cell_cost[S]     = sum_j  s[r, j] * (1 - bits[S, j]) + t[r, j] * bits[S, j]
  ```

- *Between-row cost* (vertical edge `(r-1, j) - (r, j)`):

  ```
  between_cost[S_prev, S_curr]
                    = sum_j  c_v[r-1, j] * |bits[S_prev, j] - bits[S_curr, j]|
  ```

In log space the partial partition `log_Z_r[S_curr]` evolves as

```
log_Z_0[S]      = -within_cost[0, S] - cell_cost[0, S]
log_Z_r[S_curr] = logsumexp_{S_prev}(log_Z_{r-1}[S_prev] - between_cost[r-1, S_prev, S_curr])
                  - within_cost[r, S_curr] - cell_cost[r, S_curr]
log_Z           = logsumexp_{S} log_Z_{H-1}[S].
```

`y = -tau * log Z` is the per-channel Gibbs log-partition output.

## Why use a latent grid

The chess board is `8 x 8`, but `2^8 = 256` states per row and a
`256 x 256` transition matrix is expensive. We project the trunk joint
feature to a smaller latent grid (default `H = W = 4`, so `2^W = 16`
states and a `16 x 16` transition matrix) and run the operator there.
The grid is a *latent* abstraction over the board, not literal board
squares -- it captures the same cut-bottleneck structure at a tractable
scale.

The 4x4 grid still has `2^16 = 65536` configurations, which the DP
reduces from `O(2^16)` to `O(4 * 16^2 * d_cut) = O(1024 * d_cut)`.

## Architecture-level claim

    final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * delta(y, cut_edge_energy)

where `y in R^{B, d_cut}` is the log-partition output and
`cut_edge_energy in R^{B, d_cut}` is the per-channel mean of `(c_h, c_v)`.
The gate is initialised closed (`gate_init = -2.0`) so the head starts
as a no-op.

## Falsifiers

- Primitive-level: `shuffle_logpartition` (in-batch permutation of `y`)
  must lose the slice lift.
- `uniform_edges` (replace edge costs with constant `1`) must lose the
  edge-dependent component of the lift.
- `uniform_sources` (replace source/sink penalties with constant `1`)
  must lose the source/sink dependent component of the lift.
- Architecture-level: p037 must beat i193 on its declared slice (king
  safety / fortress positions where a single open file/column matters)
  without regressing aggregate PR AUC; `shuffle_logpartition` must lose
  >=70% of that lift.

## Why this is not message passing

A `T`-step message-passing layer integrates local information through
finite-depth propagation; this operator computes the exact log-partition
over `2^(H*W)` configurations in `O(H * 2^(2W) * d_cut)`. The
computation graph is a row-wise transfer matrix product in log space,
not a `T`-step diffusion.

## Why this is not softmax pooling

Softmax pools over tokens with row-normalised weights. The cut
log-partition operator pools over the exponential family of subsets
`p(S) propto exp(-energy(S)/tau)` with an energy that includes pairwise
edge-crossing terms. The induced marginals therefore depend on global
cut structure, not on isolated token scores.

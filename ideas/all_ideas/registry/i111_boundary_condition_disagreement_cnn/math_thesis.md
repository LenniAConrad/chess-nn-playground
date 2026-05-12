# Math Thesis

Boundary-Condition Disagreement CNN

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.

Batch candidate rank: `5`.

Working thesis: Chess board edges matter -- pawns, rooks, kings, and
tactics behave differently near boundaries. A CNN's padding convention
imposes a boundary assumption. Run a shared-weight CNN under several
boundary conditions and classify from disagreement between the per-mode
feature streams.

## Multi-boundary feature decomposition

Let `x : (18, 8, 8)` be the simple_18 board tensor and let
`f_W : R^{C_in x 8 x 8} -> R^{C_out x 8 x 8}` be a single conv-norm-act
block parameterised by weights `W = (kernel, bias, GroupNorm)`. Given a
boundary condition `m` from the supported set
`{zeros, reflect, replicate, circular}`, define `f_W^{(m)}(x)` to be the
block whose padding is realised by `F.pad(x, mode=m)` followed by a
`padding=0` convolution that consumes the explicitly padded input. For
a depth-`D` trunk, the per-mode feature map is

```
F_m(x) = (f_W_D^{(m)} o ... o f_W_1^{(m)})(x)
```

where the same weight tuples `W_1, ..., W_D` are shared across all
boundary modes; only the ghost frame changes. Stacking over the
`M = |boundary_modes|` modes gives

```
F(x) : (M, channels, 8, 8).
```

## Disagreement signal

Define the per-position disagreement map

```
D(x)[c, y, x] = Var_m F(x)[m, c, y, x]
              = (1/M) sum_m (F(x)[m, c, y, x] - mean_m F(x)[m, c, y, x])^2,
```

a `(channels, 8, 8)` field that measures how much the trunk disagrees
with itself purely because of the boundary assumption. The pairwise
disagreement energies

```
P(x)[i, j] = mean_{c, y, x} (F(x)[i, c, y, x] - F(x)[j, c, y, x])^2
```

form an `(M, M)` matrix that is reported as a diagnostic.

## Classification signal

The classifier head reads, for each boundary mode `m`, the channel-wise
mean and max pool of `F_m(x)` over the 8x8 grid, plus the channel-wise
mean and max pool of `D(x)`. This gives a deterministic descriptor of
length `M * 2 * channels + 2 * channels` whose meaning is exactly the
multi-boundary disagreement decomposition described above. A small MLP
maps the descriptor to one puzzle logit.

## Distinctness

This is *not* a wavelet scattering network: there is no multi-resolution
tight frame; the boundary condition is the only thing that varies across
streams. It is *not* a multi-scale dilated CNN: every stream uses the
same kernel; only the padding ghost frame changes. It is *not* a shared
`ResearchPacketProbe` scaffold: there are no proposal-profile
diagnostics, no mechanism-family embeddings, no shared probe code. The
only signal that reaches the head is the multi-boundary disagreement
decomposition prescribed above.

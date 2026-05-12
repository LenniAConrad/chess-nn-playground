# Math Thesis

TensorSketch Interaction Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.

Batch candidate rank: `2`.

Working thesis: Some puzzle-like signals may require high-order interactions among piece-square facts. Exact high-order tuple enumeration is expensive and overlaps with Mobius/ANOVA packets, but TensorSketch can approximate polynomial-kernel interactions with a compact randomized sketch.

## Mathematical Setup

Let `x_vec ∈ R^F` be the compact board feature vector built from the flattened
`simple_18` tensor, the 12 piece-plane material counts, and per-global-plane
scalar reductions for side-to-move, castling rights and en-passant. A learned
linear projection `W: R^F -> R^D` (with `LayerNorm`) yields a base feature
vector `b = W(x_vec) ∈ R^D` (`D = base_dim`, default 512).

CountSketch is parameterized by random hashes `h: [D] -> [S]` and signs
`s: [D] -> {-1, +1}`. Its action on `b` is

```text
c(b)[k] = sum_{i: h(i) = k} s(i) * b(i),    c(b) ∈ R^S
```

A core property of CountSketch is that for any vectors `u, v ∈ R^D`,
`<c(u), c(v)>` is an unbiased estimator of `<u, v>`.

The TensorSketch construction lifts CountSketch to polynomial kernels of
degree `d`:

```text
sketch_d(b) = IFFT( FFT(c(b)) ** d )      (real part)
```

Pham & Pagh (2013) show that `sketch_d(b)` is an unbiased estimator of the
CountSketch of the `d`-fold tensor product `b^{⊗d}`, so

```text
<sketch_d(u), sketch_d(v)> ≈ <u, v>^d = K_d(u, v),
```

the homogeneous polynomial kernel of degree `d`. Concatenating
`[b, sketch_2(b), sketch_3(b)]` therefore gives an explicit randomized feature
map for the polynomial kernel `K(u, v) = α_1 <u, v> + α_2 <u, v>^2 + α_3 <u, v>^3`,
with the coefficients `α_d` learned through the per-degree log-scales and the
final MLP head. This is materially distinct from explicit high-order tuple
enumeration (e.g. Mobius / ANOVA packets) because the storage and compute
cost is `O(D + |degrees| * S)` regardless of the polynomial degree.

## Why this is honest

The hashes `(h, s)` are sampled once from a fixed `sketch_seed` and stored as
`state_dict` buffers, so the kernel approximation is deterministic across
runs. The model has no convolutional trunk, no proposal-profile diagnostics,
and no mechanism-family embeddings; the only inputs to the classifier are the
base projection, the polynomial sketches, and a small set of energy
diagnostics. All claims about high-order interactions can be falsified by the
`degree1_only` and `degree2_only` ablations defined in the source packet.

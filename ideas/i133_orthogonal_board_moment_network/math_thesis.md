# Math Thesis

Orthogonal Board Moment Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.

Batch candidate rank: `3`.

Working thesis: Puzzle-like positions may differ in global spatial moments of
piece fields: centralization, skew, diagonal concentration, king-side
imbalance, and high-order shape. Orthogonal polynomial moments provide a
compact linear-algebra descriptor that is not convolution, FFT phase
coupling, or attention.

## Moment Operator

For each learned scalar board field `F_c` computed from the simple_18 board
tensor and indexed by normalised coordinates `u, v in [-1, 1]`, the model
evaluates

```text
m_{family, c, i, j} = sum_{u, v} F_c(u, v) * basis_i(u) * basis_j(v)
```

with `family in {Legendre, Chebyshev}` and `i, j in 0..max_degree - 1`. The
family bases are the standard one-dimensional orthogonal polynomials on
`[-1, 1]`, and the two-dimensional descriptor is built from their tensor
products.

## Degree Families

Moments are bucketed by total degree `i + j`:

- `low`: `i + j <= 1` — material and global centre balance.
- `middle`: `i + j in {2, 3}` — side / wing skew, diagonal concentration.
- `high`: `i + j >= 4` — high-order spatial shape and concentration.

The middle and high groups are randomly suppressed during training (degree
dropout) so the head cannot collapse onto a single moment band.

## Why It Is Distinct

- Not FFT/bispectrum: the basis is a coordinate-polynomial integral, not a
  frequency-domain phase coupling.
- Not topology: no thresholded sublevel sets, persistent components, or Betti
  numbers.
- Not attention: the readout never queries pairs of squares with softmax
  weights — it computes fixed orthogonal projections.
- Not raw global pooling: the polynomial basis captures structured
  centralisation / skew / shape signals that mean-/max-pooling cannot
  reconstruct.

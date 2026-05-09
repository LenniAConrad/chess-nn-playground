# Math Thesis

Maxout Region Signature Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.

Batch candidate rank: `3`.

Working thesis: Puzzle-like boards may fall into distinctive piecewise-linear
activation regions.  A maxout bank can expose those regions directly by
reporting winner identities, margins, and region-transition statistics.

## Maxout regions

Given a feature vector `f(s) in R^C` at square `s in {0,...,63}` produced by a
shared convolutional trunk, a maxout bank with `J` units and `K` experts per
unit is the family of piecewise-linear maps

```
y_j(s) = max_{k=1,...,K} ( w_{j,k}^T f(s) + b_{j,k} ),    j = 1,...,J.
```

For each `(j, s)` the winning expert

```
k*_{j}(s) = argmax_{k=1,...,K} ( w_{j,k}^T f(s) + b_{j,k} )
```

partitions the input space into linear regions: two squares with the same
winning expert across all units share an identical local affine map.  The
*margin*

```
m_{j}(s) = ( w_{j, k*}^T f(s) + b_{j, k*} ) - max_{k != k*} ( w_{j,k}^T f(s) + b_{j,k} )
```

measures the local distance to the nearest decision boundary in that affine
region.  Edges of the chessboard graph along which `k*_j` changes are exactly
the region transitions of the maxout map; their density is a direct proxy for
how piecewise-linear the local feature field is.

## Region signature

The classifier ignores the raw `y_j(s)` and reads three statistics that
together identify the region structure:

1. **Winner identities.**  The histogram
   `p_{j}(k) = (1/64) * |{ s : k*_{j}(s) = k }|` describes which experts are
   used and with what frequency.  Combined with rank/file marginal histograms
   it captures coarse anisotropic region patterns on the chessboard.
2. **Margins.**  The vector `m_{j} = (m_{j}(s))_{s=0,...,63}` summarises the
   local confidence of the maxout: mean, std, max and min of `m_{j}` give a
   four-number sketch of the slack with which the affine region was chosen.
3. **Region transitions.**  With `H_{j}, V_{j}` the number of horizontal and
   vertical neighbour pairs whose winners differ, the pair `(H_{j}, V_{j})`
   counts the region-boundary crossings under axis-aligned sweeps.  These are
   the simplest discrete analogues of the total variation of `k*_{j}` and
   directly probe the geometry the thesis predicts to be discriminative.

Stacking two maxout banks lets the deeper bank's regions depend on the
shallow bank's region structure.  The classifier head therefore sees the
disjoint signatures `{p_{j}^{(b)}, m_{j}^{(b)}, H_{j}^{(b)}, V_{j}^{(b)},
\text{rank/file region counts}}_{b=1,...,B; j=1,...,J}` and a global-average
pool of the trunk to anchor the decision in the board context.  Ablations on
`(B, J, K)` and on which signature components are passed to the head are the
direct empirical tests of the thesis.

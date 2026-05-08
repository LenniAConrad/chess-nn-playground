# Math Thesis

Fisher-Geodesic Tension Network (FGTN)

Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0755_tuesday_new_york_fisher_geodesic.md`.

## Working thesis

Let `Delta^63 = { p in R^64 : p_i > 0, sum_i p_i = 1 }` be the categorical
probability simplex over the 64 squares of a chess board, equipped with
the Fisher information metric

```
g_p(u, v) = sum_i u_i v_i / p_i
```

for tangent vectors satisfying `sum_i u_i = sum_i v_i = 0`. The
square-root embedding `phi(p) = sqrt(p)` maps `Delta^63` to the positive
orthant of the unit sphere `S^63` and turns the Fisher metric into round
spherical geometry, with distance

```
d_FR(p, q) = 2 * arccos( sum_i sqrt( p_i * q_i ) ).
```

A board tensor `x in R^{C x 8 x 8}` is mapped by a learned
convolutional encoder `F_theta` into `R` source-hinge-sink paths

```
F_theta(x) = { (p_r(x), h_r(x), q_r(x)) }_{r=1..R} subset (Delta^63_eps)^{3R}
```

derived only from `x` (no engine, source, or verification metadata).
Per route, the **Fisher-Rao geodesic excess** and its directness ratio
are

```
E_r(x)   = d_FR(p_r, h_r) + d_FR(h_r, q_r) - d_FR(p_r, q_r)
rho_r(x) = E_r(x) / ( d_FR(p_r, q_r) + eps ).
```

Both are nonnegative by the triangle inequality on `Delta^63`. The
thesis is that puzzle positions tend to admit at least one route whose
hinge `h_r` does not lie on the direct Fisher geodesic between `p_r` and
`q_r`: tactical pressure passes through a sharply bent intermediate
distribution. Near-puzzle negatives may be diffuse on the simplex but
should be more geodesically aligned. The model is trained as a
supervised binary classifier with `y = (fine == 2).float()`; geometry
features are an inductive bias on the Fisher simplex, not a separate
training target.

## Puzzle-binary head

After the route head, the network forms a deterministic-length geometry
vector `geom_feat` from per-route excess, directness ratio, pairwise
Fisher-Rao distances, optional spherical hinge turns, and
gate-weighted / max aggregates of these scalars. A `LayerNorm -> MLP`
readout consumes `[pooled_board, geom_feat]` and emits the puzzle logit
`logit in R^B`. A separate geometry-only readout over `geom_feat` is
exposed as `geometry_only_logits` so the geometry-only ablation from the
markdown is available without rebuilding the model. CRTK / engine /
source metadata is reporting-only.

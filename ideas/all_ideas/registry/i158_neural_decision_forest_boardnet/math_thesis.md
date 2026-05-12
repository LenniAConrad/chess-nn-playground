# Math Thesis

Neural Decision Forest BoardNet

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.

Batch candidate rank: `4`.

Working thesis: Chess puzzle-likeness is piecewise. Different board regimes
(open vs. closed, king-attack vs. quiet, material-imbalanced vs. balanced)
are best discriminated by different cues. A differentiable decision forest
on top of a CNN feature vector models that piecewise structure with soft
oblique splits and learned leaf predictors, without a sparse expert router
and without losing end-to-end gradients.

## Trunk

A compact convolutional trunk produces a pooled board feature
`z in R^{H}` for the input simple_18 tensor. The trunk uses a stem and
`trunk_depth` residual blocks at width `trunk_width`, then concatenates
mean and max pooling and projects to the forest feature dimension
`H = hidden_dim`.

## Soft Oblique Splits

Each tree `t in {1, ..., T}` has `2^d - 1` internal nodes. Every internal
node `n` carries a learned oblique direction `w_{t,n} in R^{H}` and bias
`b_{t,n}`. Given `z`, the right-branch probability at node `n` is

```
g_{t,n}(z) = sigmoid( (w_{t,n} . z + b_{t,n}) / tau )
```

where `tau = split_temperature`. Splits are oblique because each `w_{t,n}`
is a free linear combination of the `H` board features, not an
axis-aligned indicator. They are soft because we use a sigmoid rather
than a hard threshold; the model is therefore differentiable end-to-end.

## Path-Probability Routing

Let `L = 2^d` be the number of leaves of a balanced binary tree. For
each leaf `ell` denote its path from the root by the sequence of internal
nodes `(n_0, ..., n_{d-1})` and the direction taken at each level
`(b_0, ..., b_{d-1}) in {0, 1}^d`. The probability that `z` reaches leaf
`ell` in tree `t` is the product of the per-node soft routing
probabilities along the path:

```
mu_{t, ell}(z) = prod_{k=0}^{d-1} [ b_k * g_{t, n_k}(z) + (1 - b_k) * (1 - g_{t, n_k}(z)) ]
```

By construction `sum_{ell=1}^{L} mu_{t, ell}(z) = 1` for every input `z`,
so each tree induces a soft distribution over its leaves. Each leaf
carries learned class logits `pi_{t, ell} in R^{C}` (`C = num_classes`).

## Leaf Aggregation and Forest Average

Tree `t` predicts the soft expected leaf logit:

```
y_t(z) = sum_{ell=1}^{L} mu_{t, ell}(z) * pi_{t, ell}    in R^{C}
```

The forest output is the mean over the `T` trees:

```
y(z) = (1 / T) * sum_{t=1}^{T} y_t(z)
```

For the `puzzle_binary` contract `C = 1` and the BCE-with-logits loss
operates on the squeezed scalar `y(z)`.

## Diagnostics

Because every leaf is reached with probability `mu_{t, ell}(z)`, the
model exposes per-input quantities that summarise its piecewise
behaviour:

- Leaf usage entropy
  `H_t(z) = -sum_ell mu_{t, ell}(z) * log mu_{t, ell}(z) / log L` measures
  how concentrated the routing is at one leaf vs. spread across many.
- Per-tree disagreement `Var_t y_t(z)` (variance over `T` trees) detects
  regimes where different trees specialise.
- Dominant-leaf index `argmax_ell mu_{t, ell}(z)` and probability
  `max_ell mu_{t, ell}(z)` identify which expert each sample lands on.

These are detached so they only flow into prediction artifacts, not
gradients.

## Why This Tests The Markdown Idea

The model implements three commitments the markdown packet makes:

1. *Piecewise structure*: soft routing + per-leaf logits implement
   piecewise-linear classification over a learned partition of the
   feature space, which is the explicit alternative the packet
   proposes to a sparse expert router.
2. *Oblique splits*: each `w_{t,n}` is a free direction in `R^{H}`, so
   the boundary between regimes is not constrained to a single board
   feature.
3. *No sparse routing*: the forest stays fully soft (sigmoid splits,
   product paths, mean over trees), so gradient flows through every
   parameter on every sample.

## Bespoke Implementation

This folder is a bespoke `NeuralDecisionForestBoardNet` model in
`src/chess_nn_playground/models/neural_decision_forest_boardnet.py`.
It is not a `ResearchPacketProbe` scaffold.

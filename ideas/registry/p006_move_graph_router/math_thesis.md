# Mathematical Thesis — p006 Move-Graph Router

## Operator signature

Let `X ∈ R^{B × 64 × d}` be per-square token embeddings of the simple_18
board, and let `E_b ⊆ {(i, j) : i, j ∈ {0, ..., 63}}` be the per-sample
sparse legal-move edge set, a deterministic discrete function of the
simple_18 piece planes, side-to-move, occupancy, and the precomputed
geometric attack/between-square tables.

For each source square `i` the operator computes

```
y_i = (1 / |N(i)|) · Σ_{j : (i, j) ∈ E_b} φ_θ([X_{b, i}, X_{b, j}])
```

where `φ_θ` is a shared two-layer GELU MLP and `N(i) = {j : (i, j) ∈ E_b}`
is the per-source neighbourhood. Mean-pooling over sources then yields a
single feature vector that a small MLP collapses to the primitive's
scalar logit delta `Δ`.

The combined model is the additive, gated head

```
final_logit = base_logit + σ(W_gate · trunk_pool) · Δ
```

where `base_logit` is the i193 trunk's logit and `trunk_pool` is its
joint pool feature.

## Why this is not just masked attention

`E_b` is **not** a tensor argument. It is produced inside the operator
from the discrete board state and treated with `stop_gradient`. Crucially:

1. The gradient w.r.t. the mask is zero by construction (matching the
   spec's "rules, not learned scores").
2. The computation never instantiates a dense 64×64 score matrix — the
   downstream pooling is applied to the masked edge messages directly.
3. The per-edge MLP `φ_θ` runs on `[x_i, x_j]` (concat), not on a
   bilinear or scaled-dot-product, so the head can in principle learn
   asymmetric source/target routing.

## Claimed advantage

- Information that needs to propagate along legal moves (pins, x-rays,
  battery threats) is delivered at one MGR layer rather than the
  multiple conv layers an 8x8 grid needs to discover ray semantics from
  scratch.
- The cost-per-sample is bounded by `|E_b| · d` per source where
  `|E_b| ≈ 30` for chess, so the dense-mixer worst case is
  `O(B · 64 · 64 · 2d)` for the gather step, with sparsity zeroing most
  of the contribution.

## What is actually proven

- Forward pass shape and gradient flow are unit-tested.
- Edge counts match the rule-derived adjacency for handcrafted FENs.

## What is hypothesised

- The "5% relative reduction in CRTK class-1 matched-recall FP" target
  is **not** yet measured. The scout run is gated by
  `CLAUDE_ALLOW_TRAINING=1`.
- The hidden-rebrand failure mode (operator collapsing to a dense
  64x64 mixer when all edges are present) is exposed by the
  `dense_edges` ablation; the `random_edges` ablation tests whether
  the rule structure (not just sparsity) is load-bearing.

## Failure cases

- For positions with very few legal moves (extremely closed positions),
  the per-source aggregator is averaging a handful of edges and may be
  noisy. The aggregation uses `degree.clamp_min(1.0)` so the operator
  is well-defined even for source squares with no outgoing edges.
- Pathological side-to-move encoding mis-reads in the simple_18 tensor
  would flip the own/enemy mask; the operator inherits this risk from
  the trunk's encoding.

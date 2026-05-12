# Math Thesis

Sparse Expert Board Router

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.

Batch candidate rank: `5`.

Working thesis: Chess positions are heterogeneous. Endgames, king attacks, pawn races, blocked centers, and material imbalances may need different feature extractors. A sparse mixture of small board experts can route positions to specialized encoders without requiring a giant monolithic model.

## Formal Object

Let `x ∈ R^{18×8×8}` denote a `simple_18` board tensor, and let `s(x) ∈ R^R` denote a deterministic routing summary (material totals, king centroids, coarse-quadrant occupancy means, side-to-move). A small CNN stem `φ` produces a spatial summary whose mean and max pool are concatenated with `s(x)` and layer-normalised:

```text
r(x) = LayerNorm([s(x), pool(φ(x))]) ∈ R^D
```

A router `g_θ : R^D → R^E` produces logits over `E=6` experts. Sparse selection keeps the top `k=2` logits via masked softmax:

```text
g_topk(x) = softmax( masktopk_k( g_θ(r(x)) ) )
```

Each expert `f_i` is a small board encoder producing a hidden vector `h_i ∈ R^H`. The mixture hidden state, mixture logit, and fused logit are:

```text
h_mix(x) = Σ_i  g_topk(x)_i · h_i(x)
ℓ_mix(x) = Σ_i  g_topk(x)_i · w_i^T h_i(x)        # per-expert binary heads
ℓ_fuse(x) = u^T h_mix(x)                            # fused MLP head
ℓ(x)     = σ(α) · ℓ_mix(x) + (1 - σ(α)) · ℓ_fuse(x) # learned blend
```

## Auxiliary Losses

Two regularisers from the source packet are exposed as diagnostics so the trainer can attach them with arbitrary weights:

```text
L_balance(B) = E · Σ_i ( mean_b g_topk(x_b)_i - 1/E )^2
             + E · Σ_i ( select_fraction_i · mean_b router_probs(x_b)_i )
L_entropy(B) = - mean_b H( router_probs(x_b) )
```

The first term is a hybrid of the expected-mass squared-deviation and a Switch-Transformer style auxiliary using selection counts; the second penalises overly-deterministic routing.

## Why It Is Distinct

- Selection is over computations (experts) rather than gating a single CNN, distinguishing it from piece-conditioned hypernetworks.
- Experts are jointly trained with a sparse top-k router, distinguishing it from ensemble baselines.
- The router consumes both deterministic chess summaries and a small spatial CNN summary so the `material_only` ablation removes only the spatial half.

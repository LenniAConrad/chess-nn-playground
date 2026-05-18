# Math Thesis

Submodular Coverage Bottleneck

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.

Batch candidate rank: `5`.

## Setting

Let `a ∈ [0, 1]^M` be sigmoid concept activations from four board sources (patch, line, king-ring, material) and `W ∈ R^{M × K}_{≥0}` be a learned nonnegative coverage matrix. Define covered attributes

```
c_k = 1 - ∏_i (1 - a_i W_{i,k})        ∈ [0, 1]
```

and a coverage score

```
F(a) = Σ_k β_k c_k,
```

with per-attribute saliences `β ∈ R^K`. Because `W ≥ 0`, `c_k` is monotone in every `a_i` and the set function induced by thresholding `a` is submodular: each repeated activation of the same covered attribute contributes a strictly smaller increment than the previous one.

## Marginal gain (closed form)

For each concept `i`,

```
F(a) - F(a \ {i}) = Σ_k β_k (c_k - c_k^{-i})
                  = Σ_k β_k · exp(Σ_j log(1 - a_j W_{j,k})) · (a_i W_{i,k}) / (1 - a_i W_{i,k}).
```

This lets the model expose all `M` marginal gains per forward pass without re-running the coverage layer `M` times.

## Head and diagnostics

The puzzle head reads `[F(a), c, top-T marginal gains, H(a)]` with concept entropy `H(a) = -Σ_i a_i log a_i + (1 - a_i) log(1 - a_i)`. Diagnostics expose the coverage vector `c`, top marginal gains, the additive-pool counterfactual `Σ_i a_i W_{i,k}`, the saturation gap between coverage and the additive pool, the active-concept count, and the maximum marginal gain.

## What the ablations falsify

- `additive_pool` removes diminishing returns; if the head is no worse, the coverage non-linearity is unnecessary.
- `no_marginal_gains` removes the marginal-gain head input; if the head is no worse, only covered attributes matter.
- `unconstrained_W` allows signed weights; if the head is better, the submodular monotonicity constraint may be too restrictive.
- `random_concepts` freezes the concept encoders; if the head is unchanged, the coverage head is doing all the work.
- `material_concepts_only` keeps only material concepts; if the head is strong, the model is shortcutting to material balance rather than learning tactical concepts.

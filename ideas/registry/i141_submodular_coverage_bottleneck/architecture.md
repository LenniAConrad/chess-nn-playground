# Architecture

`Submodular Coverage Bottleneck` is a bespoke classifier whose head reads from a differentiable product-saturation coverage function over learned concept activations rather than from an ordinary additive pool.

## Implementation Binding

- Registered model name: `submodular_coverage_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/trunk/submodular_coverage_bottleneck.py`
- Idea-local wrapper: `ideas/registry/i141_submodular_coverage_bottleneck/model.py`

- Mechanism family: `convex` (submodular set function).
- Input: simple_18 board tensor only; CRTK/source metadata is reporting-only.

## Concept activations

Four concept sources are extracted from the board tensor and concatenated:

- `patch` — a two-block convolutional stem followed by 2x2 average pooling produces 16 spatial concepts (one per 4x4 patch).
- `line` — per-rank, per-file, and diagonal occupancy summaries (30 inputs) feed an MLP that emits `num_line_concepts` logits.
- `king` — king-centred Chebyshev rings (radius 1 and 2, one per side, four inputs) feed an MLP that emits `num_king_concepts` logits.
- `material` — material / side-to-move / castling / en-passant counts (25 inputs) feed an MLP that emits `num_material_concepts` logits.

Logits from all sources are concatenated and squashed by `sigmoid` to give `a ∈ [0, 1]^M`.

## Coverage layer

A learned coverage matrix `W ∈ R^{M × K}` is held as `softplus` of an unconstrained parameter, enforcing `W_{i,k} ≥ 0` (the submodular regime). With nonnegative weights the covered attributes

```
c_k = 1 - ∏_i (1 - a_i W_{i,k})
```

are monotone in each `a_i` and submodular in the set of active concepts — the second copy of the same concept adds less than the first. Per-attribute saliences `β ∈ R^K` give the coverage score

```
F(a) = Σ_k β_k c_k.
```

Marginal gains `gain_i = F(a) - F(a \ {i})` are computed in closed form from

```
c_k - c_k^{-i} = exp(Σ_j log(1 - a_j W_{j,k})) · (a_i W_{i,k}) / (1 - a_i W_{i,k}),
```

so no per-concept ablation pass is needed.

## Head

The classifier consumes

```
[F(a), c, top-T marginal gains, concept entropy H(a)]
```

with `H(a) = -Σ_i a_i log a_i + (1 - a_i) log(1 - a_i)`. It is a LayerNorm + Linear + GELU MLP that returns one puzzle logit and the following diagnostics: `coverage`, `coverage_score`, `marginal_gains`, `top_marginal_values`, `top_marginal_indices`, `concept_entropy`, `active_concept_count`, `coverage_energy`, `additive_pool_energy`, `saturation_gap`, `max_marginal_gain`, `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`, `submodular_coverage_ablation`, `submodular_concept_total`, `submodular_attribute_total`.

This is materially distinct from the shared `ResearchPacketProbe` scaffold (no coverage matrix, no marginal-gain head), from prototype/dictionary networks (coverage is a set function with diminishing returns rather than a sparse distance to anchors), from attention (no normalized weighted value pooling), and from mixture-of-experts (no sparse routing).

## Supported ablations

`SubmodularCoverageBottleneckNetwork.ABLATIONS` enumerates the testable variants:

- `none` — full implementation as described above.
- `additive_pool` — replace `c_k` with the additive sum `Σ_i a_i W_{i,k}` (no diminishing returns) and feed that linear pool to the head. Tests whether saturation matters.
- `no_marginal_gains` — keep the coverage layer but zero the marginal-gain features in the head input. Tests whether marginal structure carries signal beyond `F(a)` and `c`.
- `unconstrained_W` — drop the `softplus` nonnegativity constraint so `W` can take any sign. Tests whether the submodular monotonicity constraint matters.
- `random_concepts` — freeze the concept encoders at initialization so only the coverage layer and head can learn. Tests whether the learned concept structure carries signal.
- `material_concepts_only` — keep only the material/count concept source and zero the patch/line/king activations. Tests whether the model shortcuts to material balance.

# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/submodular_coverage_bottleneck.py`.
- Idea-local wrapper: `ideas/registry/i141_submodular_coverage_bottleneck/model.py`.
- Registry key: `submodular_coverage_bottleneck`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.
- Batch candidate: `Submodular Coverage Bottleneck`.

## Wiring

- Coverage matrix `W` is held as `softplus(coverage_logits)` so `W ≥ 0` by construction. The `unconstrained_W` ablation skips the `softplus` so the head can use signed weights.
- The per-attribute salience `β` is an unconstrained parameter; sign is not restricted.
- Marginal gains use a numerically stable expression in `(B, M, K)` space: `gain_{b,i} = Σ_k β_k · exp(sum_log_{b,k}) · (a_{b,i} W_{i,k}) / (1 - a_{b,i} W_{i,k})`, where `sum_log = Σ_i log(1 - a W)` is computed once per batch.
- `(1 - a_i W_{i,k})` is clamped to a small positive floor before taking `log` to avoid `-inf` when an activation saturates.
- The `material_concepts_only` ablation masks the patch/line/king activation slice to zero before the coverage layer runs, so those concepts never contribute.
- The `random_concepts` ablation freezes the patch CNN and the line/king/material MLP heads via `requires_grad_(False)`; only the coverage matrix, saliences, and classifier MLP receive gradients.
- `top_marginal` controls how many sorted marginal-gain values feed the head; padding fires only when `M < top_marginal`, which is degenerate.

## Inputs and shape contracts

- Only the simple_18 encoding and the one-logit `puzzle_binary` contract are supported. Both constraints are enforced at construction time.
- Concept counts default to 16 patch + 12 line + 8 king + 8 material = 44 total concepts; with 16 attributes the coverage tensor `a W` is `(B, 44, 16)`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.

## Diagnostics

Forward returns the puzzle logit plus: `coverage`, `coverage_score`, `marginal_gains`, `top_marginal_values`, `top_marginal_indices`, `concept_entropy`, `active_concept_count`, `coverage_energy`, `additive_pool_energy`, `saturation_gap`, `max_marginal_gain`, `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`, `submodular_coverage_ablation`, `submodular_concept_total`, `submodular_attribute_total`. These match the keys other implemented ideas surface so the shared prediction-artifact writer can consume them without changes.

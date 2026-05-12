# Architecture

`Tactical Symptom Bayesian Network` is a board-only puzzle_binary
architecture that forces the classifier to combine differentiable
*noisy logical symptoms* through a small Bayesian-style symptom
network instead of an uninterpretable dense readout. The thesis (see
`math_thesis.md`) is that puzzle-vs-non-puzzle structure is a
conjunction of noisy symptoms — *king exposed*, *defender overloaded*,
*line opened*, *piece pinned*, *queen aligned*, *escape squares
reduced*, *target under-defended* — so the puzzle classifier should
combine them with noisy-AND/noisy-OR aggregation rather than free
pooling.

The model is bespoke board-only: it consumes the repository
`simple_18` current-board tensor `(B, 18, 8, 8)` and returns one
puzzle logit for the BCE-with-logits `puzzle_binary` trainer.

## Mechanism

1. **Conv trunk.** A compact `BoardFeatureTrunk` (Conv → BN/GroupNorm
   → GELU → optional Dropout2d, repeated `depth` times) produces a
   per-square feature map `(B, channels, 8, 8)`.

2. **Symptom heads.** A `1x1` conv readout produces `K = symptoms`
   per-square sigmoid maps. The image-level symptom probability is a
   noisy-OR over the 64 squares so that a single strong square is
   sufficient to fire the symptom:

   ```text
   s_k_sq = sigmoid(symptom_head_k(board_features)[sq])
   s_k    = 1 - prod_sq (1 - s_k_sq)
   ```

3. **Noisy-OR cause layer.** The K symptoms feed `J = latent_causes`
   noisy-OR causes parameterised by sigmoid weights `w_jk in [0, 1]`
   plus a per-cause leak probability `leak_j`:

   ```text
   cause_j = 1 - (1 - leak_j) * prod_k (1 - w_jk * s_k)
   ```

4. **Noisy-AND-OR aggregator.** A learned mixture of two complementary
   differentiable aggregations of the J causes:

   ```text
   prob_or  = 1 - prod_j (1 - or_w_j  * cause_j)
   prob_and = prod_j (1 - and_w_j * (1 - cause_j))
   alpha       = sigmoid(mix_logit)
   puzzle_prob = alpha * prob_or + (1 - alpha) * prob_and
   ```

5. **Residual logit + output.** A small MLP residual head reads
   `(features.mean, features.amax, symptoms, causes)` and produces an
   additive `residual_logit`. The puzzle output follows the source
   packet's recombination rule

   ```text
   logits = logit(clamp(puzzle_prob)) + residual_weight * residual_logit
   ```

   where `residual_weight` is a learned scalar initialised from
   `residual_weight_init` (config default `0.1`, matching the source
   packet's `residual_weight_init`).

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
`puzzle_binary` BCE-with-logits trainer (`num_classes == 1`), plus
diagnostics:

- `logits`, `puzzle_prob`, `evidence_logit`, `residual_logit`,
  `residual_weight`, `symptom_linear_logit`, `symptom_max`,
  `symptom_mean`, `symptom_entropy`, `cause_max`, `cause_mean`,
  `cause_entropy`, `noisy_or_prob`, `noisy_and_prob`, `and_or_alpha`,
  `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`: `(B,)`.

## Ablations

The constructor accepts the three ablations required by the source
packet plus the default:

- `none` — the full noisy-AND/noisy-OR symptom network.
- `linear_symptom_readout` — replaces the noisy logical combination
  with a linear readout of the symptom probabilities. Tests whether
  the noisy logical structure helps over a flat linear classifier on
  the same symptom basis.
- `no_residual_logit` — drops the residual logit so the output is
  exactly `logit(puzzle_prob)`. Tests the pure symptom bottleneck.
- `symptom_dropout` — applies `symptom_dropout` Bernoulli dropout to
  the K symptom probabilities during training. Tests robustness of
  the symptom decomposition.

## Implementation Binding

- Registered model name: `tactical_symptom_bayesian_network`
- Source implementation file: `src/chess_nn_playground/models/tactical_symptom_bayesian_network.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i194_tactical_symptom_bayesian_network/model.py`

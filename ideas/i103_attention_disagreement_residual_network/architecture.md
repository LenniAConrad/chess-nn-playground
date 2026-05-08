# Architecture

`Attention Disagreement Residual Network` (ADRN) builds square tokens from the
current-board `simple_18` tensor and runs `F` independent learned query banks
with `Q` queries each over a shared key/value projection of those tokens. The
classifier reads the mean attended value plus residual disagreement statistics
across the families, so puzzle-likeness is decided from the *spread* of
attention interpretations rather than from a single attention pattern.

## Tensor Contract

```text
input:                (B, 18, 8, 8)
square tokens:        (B, 64, D)
attention maps:       (B, F, Q, 64)
attended values:      (B, F, Q, D)
disagreement vector:  (B, 2D + F*(F-1)/2 + F + 4)
logits:               (B,)
```

## Components

- Square tokenizer: per-square MLP over channels concatenated with deterministic
  rank/file/centred coordinates, edge distance, and square colour. No engine,
  search, source, or CRTK metadata participates.
- Independent query banks: `nn.Parameter` of shape `(F, Q, D)`. The shared
  query projection `W_q`, key projection `W_k`, and value projection `W_v` are
  applied to every family so that disagreement reflects query content, not
  separate value subspaces.
- Per-family attention: `softmax(W_q q_{f,i} . W_k t_n / sqrt(D))` over the 64
  square tokens, giving `A_f in (B, F, Q, 64)`.
- Disagreement summaries:
  - Per-family attended value mean and the per-family residual standard
    deviation across the family axis (the attended residual itself).
  - Pairwise Jensen-Shannon divergence between family-averaged attention
    distributions.
  - Per-family normalised attention entropy mean, plus the variance of those
    family entropies.
  - Maximum cosine distance between family-averaged attention distributions
    (attention residual on the simplex).
  - Maximum cosine distance between per-query attention distributions taken
    across families (cross-family query-map disagreement).
  - Attended-value covariance trace summarising the spread of attended
    coordinates across families.
- Classifier head: LayerNorm + linear + GELU + dropout + linear, outputting one
  puzzle logit.

## Output Diagnostics

Forward returns `logits` plus diagnostics used for ablation and reporting:
`attention`, `attended_values`, `disagreement_features`,
`attention_js_divergence_mean`, `attention_js_divergence_max`,
`attention_entropy_variance`, `attention_entropy_mean`,
`family_map_cosine_distance_mean`, `family_map_cosine_distance_max`,
`query_map_cosine_distance_max`, `attended_residual_norm`,
`attended_covariance_trace`, and `attended_mean_norm`.

## Implementation Binding

- Registered model name: `attention_disagreement_residual_network`.
- Source implementation file: `src/chess_nn_playground/models/attention_disagreement_residual_network.py`.
- Idea-local wrapper: `ideas/i103_attention_disagreement_residual_network/model.py`.

# Architecture

`Attention Perturbation Sensitivity Network` (APSN) treats attention as a
hypothesis selector, not as the puzzle decision itself. A small attention
reader produces per-query attention maps over the 64 board squares and a base
latent. Four deterministic mask families - top-attention, low-attention
occupied, permutation-random occupied, and the 3x3 neighbourhood of the
top-attention square - zero the 12 piece planes at the selected squares. The
shared encoder is re-run on each masked variant and the puzzle classifier
reads the base latent together with sensitivity contrasts
``||z(x) - z(mask_*(x))||`` and a small set of attention diagnostics. No
engine, search, source, or CRTK metadata enters the forward pass.

## Tensor Contract

```text
input:                             (B, 18, 8, 8)
square tokens:                     (B, 64, D)
per-query attention:               (B, Q, 64), softmax over squares
masked variants (top/low/rand/nbhd): (B, 18, 8, 8) each
variant latents (z(x), z_top, ...):  (B, D) each
sensitivity scalars (4):           (B,)
sensitivity contrasts (4):         (B,)
attention diagnostics (7):         (B,)
logits:                            (B,)
```

`D = token_dim` and `Q = num_queries`. Padded boards are not used; sensitivity
is computed across whole-board reruns.

## Components

- Square tokenizer: per-square features concatenate the 18 board planes with
  six deterministic coordinates (rank, file, centred rank, centred file, edge
  distance, square colour) and pass through a two-layer MLP with `LayerNorm`
  and `GELU` to produce token embeddings of dimension `token_dim`.
- Attention reader: `Q` learnable query vectors form the queries; the tokens
  produce keys and values via independent linear projections. The attention
  map ``A in R^{Q x 64}`` is softmaxed over squares and the base latent is the
  per-query mean of attended values, normalised by `LayerNorm`.
- Mask construction (deterministic, gradients are detached on the index
  selection):
  - `keep_top` zeros the top-K squares of the per-query mean attention.
  - `keep_low_occupied` zeros the K lowest-attention occupied squares, where
    occupancy is the union of the 12 piece planes.
  - `keep_random_occupied` zeros K occupied squares chosen by a fixed seeded
    permutation of the 64 squares - independent of attention - so that the
    central ablation `random_mask_sensitivity` can be reproduced just by
    swapping which mask drives the contrast.
  - `keep_top_neighborhood` zeros the 3x3 board neighbourhood of the
    top-attention square (computed from a fixed 64x64 adjacency buffer).
  Each mask zeros the 12 piece planes at the selected squares while leaving
  the 6 global planes (side-to-move, castling, en-passant) untouched.
- Sensitivity head: the same encoder is re-run on each masked board to
  produce `latent_top`, `latent_low`, `latent_random`, and
  `latent_neighborhood`. Sensitivities are
  `delta_* = ||latent_base - latent_*||_2`, with the four contrasts
  `contrast_top_minus_low`, `contrast_top_minus_random`,
  `contrast_neighborhood_minus_top`, and `ratio_top_over_low`.
- Attention diagnostics: per-query entropy (normalised by `log 64`), peak
  attention, top-K attention mass, occupied vs empty mass, query-axis variance
  (a disagreement signal), and per-square max-minus-min range.
- Classifier: `LayerNorm` + `Linear(head_hidden)` + `GELU` + dropout +
  `Linear(1)` over `latent_base` concatenated with the eight sensitivity
  features and seven attention diagnostics. Returns one puzzle logit.

## Output Diagnostics

Forward returns `logits` plus `latent_base`, `latent_top`, `latent_low`,
`latent_random`, `latent_neighborhood`, `attention`, `per_square_attention`,
`mean_query_entropy`, `max_attention`, `topk_attention_mass`,
`attention_occupied_mass`, `attention_empty_mass`,
`attention_query_disagreement`, `attention_range`, `occupancy_mask`,
`keep_mask_top`, `keep_mask_low`, `keep_mask_random`,
`keep_mask_neighborhood`, `delta_top`, `delta_low`, `delta_random`,
`delta_neighborhood`, `contrast_top_minus_low`, `contrast_top_minus_random`,
`contrast_neighborhood_minus_top`, `ratio_top_over_low`,
`sensitivity_features`, and `attention_features`.

## Implementation Binding

- Registered model name: `attention_perturbation_sensitivity_network`.
- Source implementation file: `src/chess_nn_playground/models/attention_perturbation_sensitivity_network.py`.
- Idea-local wrapper: `ideas/i106_attention_perturbation_sensitivity_network/model.py`.

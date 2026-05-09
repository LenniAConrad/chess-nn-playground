# Architecture

`Vector-Quantized Motif Codebook Net` realises the source packet's
"learned discrete codebook of board motifs" thesis as a bespoke
architecture for the repo's `puzzle_binary` task. A compact
convolutional encoder produces a per-square embedding map, a
vector-quantization layer routes every square to a learned codebook
entry, and a small MLP head reads pooled quantized features, the
batch code-usage histogram, and a learned spatial code-map
embedding.

## Implementation Binding

- Registered model name: `vector_quantized_motif_codebook_net`
- Source implementation file: `src/chess_nn_playground/models/vector_quantized_motif_codebook_net.py`
- Idea-local wrapper: `ideas/i159_vector_quantized_motif_codebook_net/model.py`

## Modules

`VectorQuantizedMotifCodebookNet` accepts the project's
`(B, 18, 8, 8)` board tensor only. CRTK / source / engine /
verification metadata is reporting-only and is not consumed.

1. **Board motif encoder.** `BoardMotifEncoder` runs `depth` blocks of
   `3x3 Conv2d -> [BatchNorm2d ->] GELU -> [Dropout2d]` at width
   `channels`, then a `1x1 Conv2d` that projects to the codebook
   dimension `code_dim`. The output is `z_e` of shape
   `(B, code_dim, 8, 8)`: one motif vector per square.
2. **Motif codebook quantizer.** `MotifCodebookQuantizer` holds a
   `(num_codes, code_dim)` codebook `C`. Per-square distances to the
   codebook are computed as
   `||z_e[b,:,i,j] - C[k]||^2 = ||z_e||^2 - 2 z_e . C[k] + ||C[k]||^2`,
   and each square selects the nearest entry
   `k*(b,i,j) = argmin_k ||z_e[b,:,i,j] - C[k]||^2`. The quantized
   feature map is `z_q[b,:,i,j] = C[k*(b,i,j)]`. Gradients flow back
   into the encoder via the straight-through estimator
   `z_q_st = z_e + stop_grad(z_q - z_e)`. The codebook itself is
   updated by exponential moving averages of cluster usage
   `N_k <- decay * N_k + (1 - decay) * sum_t 1{k*=k}` and centroid
   sums `m_k <- decay * m_k + (1 - decay) * sum_t z_e_t 1{k*=k}`,
   with Laplace-smoothed centroids
   `C[k] = m_k / ((N_k + epsilon) / (sum_j N_j + K * epsilon) * sum_j N_j)`.
   Commitment and codebook MSE losses are returned as diagnostics
   (the BCE-with-logits trainer does not need them, but they are
   available for sweeps that want to add them as an auxiliary
   objective).
3. **Motif inventory head.** The classifier reads three views:
   - global mean and max pools of `z_q` over the 8x8 board
     (`2 * code_dim` features),
   - a per-batch code-usage histogram over the 64 squares
     (`num_codes` features), and
   - a learned `(num_codes, hidden_dim)` `code_map_embedding`
     evaluated at every square's selected code and averaged over the
     board (`hidden_dim` features).
   These are concatenated, normalised by `LayerNorm`, and passed
   through `Linear -> GELU -> [Dropout] -> Linear` to produce the
   puzzle logit (`(B,)` for `num_classes == 1`).

## Codebook Mathematics

For each square `(i, j)` of input `x` and encoder output
`z_e = encoder(x)`:

```
k*(b,i,j) = argmin_k || z_e[b,:,i,j] - C[k] ||^2
z_q[b,:,i,j] = C[k*(b,i,j)]
z_q_st       = z_e + stop_grad(z_q - z_e)        (STE)
L_commit     = mean_b,i,j || z_e - stop_grad(z_q) ||^2
L_codebook   = mean_b,i,j || stop_grad(z_e) - z_q ||^2
```

EMA codebook update (training only, no gradient):

```
N_k       <- decay * N_k       + (1 - decay) * sum_{b,i,j} 1{k*(b,i,j) = k}
m_k       <- decay * m_k       + (1 - decay) * sum_{b,i,j} 1{k*(b,i,j) = k} * z_e[b,:,i,j]
N_k_smooth = (N_k + eps) / (sum_j N_j + K * eps) * sum_j N_j
C[k]      <- m_k / N_k_smooth
```

Code usage probability per sample is
`p_k(b) = (1 / 64) * sum_{i,j} 1{k*(b,i,j) = k}`, and the per-sample
codebook entropy is `H(b) = - sum_k p_k(b) log p_k(b)` with
perplexity `exp(H(b))`.

## Loss

The default trainer wires standard BCE-with-logits on
`output["logits"]`. Gradient flows through the encoder via the
straight-through estimator on `z_q_st`, through the
`code_map_embedding`, and through the head. The codebook itself is
trained without gradients via the EMA updates above. The
commitment and codebook losses are returned in the output dict so
sweeps can add them as auxiliary losses if desired; this base
specification does not require them.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)` for `num_classes == 1` (BCE-compatible
  log-odds), `(B, num_classes)` otherwise.
- `prob`: shape `(B,)` sigmoid probability when `num_classes == 1`.
- `code_usage_entropy`: shape `(B,)`, per-sample Shannon entropy of
  the 64-square code histogram (nats).
- `code_perplexity`: shape `(B,)`, `exp(code_usage_entropy)`.
- `active_codes`: shape `(B,)`, count of codebook entries used at
  least once on the board.
- `mean_quantization_distance`: shape `(B,)`, mean nearest-neighbour
  squared distance to the codebook over the 64 squares.
- `commitment_loss`: shape `(B,)`, broadcast scalar of the
  encoder-side STE loss.
- `codebook_loss`: shape `(B,)`, broadcast scalar of the
  codebook-side STE loss.
- `dominant_code_probability`: shape `(B,)`, maximum per-code
  frequency among the 64 squares.
- `spatial_code_homogeneity`: shape `(B,)`, fraction of squares that
  match the top-left square's code.
- `encoder_feature_energy`: shape `(B,)`, mean square of `z_e`.
- `quantized_feature_energy`: shape `(B,)`, mean square of `z_q`.
- `code_map`: shape `(B, 8, 8)`, integer codebook indices per square.

## Contract

- Input: `(B, 18, 8, 8)` simple_18 board tensor only. Engine,
  verification, source, CRTK, principal-variation, mate-score, and
  best-move metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  `puzzle_binary` BCE-with-logits trainer, plus the diagnostics
  listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`;
  fine label `2` maps to binary target `1`.

# Architecture

`Slot Attention Role Binding Network` (SARBN) extracts up to 32 occupied piece
tokens from the current-board `simple_18` tensor and runs `T = 3` slot-attention
iterations that softly bind those pieces to `S = 8` learned latent role slots.
The puzzle classifier reads the final slot vectors plus assignment-entropy,
slot-mass, and per-iteration update-residual diagnostics. No engine, search,
source, or CRTK metadata enters the forward pass.

## Tensor Contract

```text
input:                       (B, 18, 8, 8)
piece tokens (padded to 32): (B, 32, F)
token mask (occupied flag):  (B, 32)
slot bank:                   (B, S, D), default S = 8
assignment per iteration:    (B, T, S, 32)
slot updates per iteration:  (B, T, S, D)
slot update residual norms:  (B, T)
logits:                      (B,)
```

`F` is the per-square feature dimension before encoding (12 piece-plane indicators,
6 global planes, and 6 deterministic coordinates) and `D` is the slot embedding
dimension. Padded token positions carry zeros and a token-mask of 0 so they cannot
contribute to slot assignment.

## Components

- Token extractor: per-square features are formed by concatenating the 12 piece
  one-hot planes, the 6 global planes (side-to-move, four castling-right planes,
  en-passant), and a 6-dimensional deterministic coordinate code (rank, file,
  centred rank, centred file, edge distance, square colour). Occupied squares
  are selected in deterministic rank-major order to give up to 32 tokens; padded
  positions receive zero features and a zero token mask.
- Token encoder: a two-layer MLP with `LayerNorm` + `GELU` projects each token
  into a `token_dim`-d embedding. Token embeddings are masked again so padding
  cannot leak through key/value projections.
- Slot bank: `S` learnable slot prototypes `mu_s` with per-dimension log-sigmas.
  At training time, slots are sampled as `slot_s = mu_s + sigma_s * eps`; at
  evaluation, slots are deterministic at `mu_s`.
- Slot attention iteration (`T = 3`):
  - `q_s = W_q LayerNorm(slot_s) / sqrt(D)`
  - `k_p = W_k LayerNorm(token_p)`, `v_p = W_v LayerNorm(token_p)`, both masked
    to padded positions.
  - `attention_{s, p} = softmax_s(q_s . k_p)` (softmax over slots, so each
    piece's mass is softly partitioned across role slots) with masked positions
    forced to zero contribution.
  - `weights_{s, p} = attention_{s, p} / sum_p attention_{s, p}` and
    `update_s = sum_p weights_{s, p} v_p`.
  - `slot_s <- GRUCell(update_s, slot_s)` followed by a residual MLP
    `slot_s <- slot_s + MLP(LayerNorm(slot_s))`.
  - The L2 norm of the slot update `slot_s_new - slot_s_prev` is logged per
    iteration as `update_residuals`.
- Diagnostics derived from the final assignment:
  - `slot_mass` (B, S) and `slot_share` (B, S) (mass divided by the number of
    real tokens) as a per-slot competition signal.
  - `slot_self_entropy` (B, S) over the 32 token slots and per-token entropy
    `mean_token_entropy`, `token_entropy_variance` over the slot axis after
    column-renormalisation, summarising assignment shape.
  - `slot_norms`, `slot_dispersion` (mean off-diagonal cosine of slot vectors),
    and the norm of `slot_updates` averaged and maxed across iterations.
- Classifier: `LayerNorm` + `Linear(head_hidden)` + `GELU` + dropout +
  `Linear(1)` over `flatten(slots)` concatenated with the diagnostic feature
  vector. Returns one puzzle logit.

## Output Diagnostics

Forward returns `logits` plus `slots`, `assignments`, `slot_updates`,
`update_residuals`, `slot_mass`, `slot_share`, `slot_self_entropy`,
`per_token_entropy`, `mean_token_entropy`, `token_entropy_variance`,
`slot_norms`, `slot_dispersion`, `token_mask`, `occupancy_mask`, and the pooled
`diagnostic_features` vector.

## Implementation Binding

- Registered model name: `slot_attention_role_binding_network`.
- Source implementation file: `src/chess_nn_playground/models/trunk/slot_attention_role_binding_network.py`.
- Idea-local wrapper: `ideas/registry/i105_slot_attention_role_binding_network/model.py`.

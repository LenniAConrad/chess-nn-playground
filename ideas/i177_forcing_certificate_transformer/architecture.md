# Architecture

`Forcing-Certificate Transformer` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one
puzzle logit per position by trying to assemble a small,
slot-structured tactical certificate
(`attacker / target / defender / blocker / tempo`) instead of pooling
a single global board embedding.

## Mechanism

A compact convolutional stem turns the 18-plane board into a per-square
feature map, which is flattened into 64 square tokens. Each token is
projected to width `token_dim` and combined with a learnable per-square
positional embedding plus geometric coordinates (rank, file, centred
rank/file, edge distance, square colour) so the slots can address
specific squares.

A bank of `num_slots` learnable certificate-slot queries cross-attends
to the 64 square tokens. The cross-attention scores are biased by a
per-slot mixture of fixed chess relation matrices (`same_rank`,
`same_file`, two diagonals, knight reach, king-adjacent, and the two
directional pawn attack patterns). Concretely each slot `k` learns an
anchor distribution `α_k ∈ Δ(64)` and a mixing vector `mix_k ∈ R^R` so
the bias from slot `k` to token `j` is

```
bias[k, j] = sum_r mix[k, r] * sum_i α_k[i] * R_r[i, j]
```

This is the packet's `relation_bias` between slot anchors (`attacker`,
`defender`, `blocker`, ...) and target squares.

Each slot is updated for `slot_iters` iterations:

```
slot_k <- slot_k + cross_attention(slot_k, square_tokens, relation_bias)
slot_k <- slot_k + slot_mlp(slot_k)
```

After the final iteration each slot emits a scalar via a small head:

```
slot_score_k = MLP(slot_k)
puzzle_logit = logsumexp(slot_score_k) + global_residual_logit
```

The `global_residual_logit` is read from a mean-pooled MLP over the
square tokens, exactly as the packet specifies.

Inputs to the model are limited to the `simple_18` board tensor.
Engine, verification, source, and CRTK metadata are never used.

## Trunk and tokenizer

A stack of `depth` `Conv3x3 → BatchNorm → ReLU` layers turns the
18-plane board into a per-square feature map of width `channels`. Each
square's feature vector is concatenated with 6 fixed coordinate
features and projected through `LayerNorm → Linear` to width
`token_dim`. A learnable 64-row positional embedding of width
`token_dim` is added so the slots can identify specific squares.

## Certificate slots

`num_slots` learnable queries of width `token_dim` are used as the
certificate slots. They cross-attend to the 64 square tokens with
`num_heads` attention heads. The relation-bias term described above is
broadcast across heads and added to attention logits before softmax.
After each cross-attention layer a residual `LayerNorm → Linear → GELU
→ Dropout → Linear` block refines the slot embeddings. The default
configuration uses `slot_iters = 2`.

## Readout

The slot score head is `LayerNorm → Linear → GELU → Dropout → Linear`,
mapping each slot embedding to a scalar `slot_score_k`. The
certificate logit is `logsumexp_k slot_score_k`. A separate global
residual head reads the mean-pooled square tokens through the same
MLP recipe and emits `global_residual_logit`. The final puzzle logit
is

```
puzzle_logit = logsumexp(slot_scores) + global_residual_logit
```

When `num_classes > 1` the puzzle logit is written into the last
column of a zero-padded logits tensor so the BCE-with-logits trainer
contract still holds.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the repository `puzzle_binary` BCE-with-logits trainer. All
tensors are finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` when
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `slot_scores`: `(B, num_slots)` per-slot scalar scores.
- `slot_logsumexp`: `(B,)` log-sum-exp of `slot_scores`.
- `global_residual_logit`: `(B,)` residual logit from the mean-
  pooled MLP head.
- `slot_attention`: `(B, num_slots, 64)` slot attention map after
  the final cross-attention layer.
- `slot_attention_entropy`: `(B, num_slots)` per-slot attention
  entropy normalised by `log(64)`.
- `slot_attention_entropy_mean`: `(B,)` mean attention entropy.
- `slot_attention_max`, `slot_attention_margin`: `(B, num_slots)`
  attention concentration diagnostics.
- `slot_diversity`: `(B,)` mean pairwise total-variation distance
  between slot attention maps -- the packet's slot diversity signal.
- `slot_features`: `(B, num_slots, token_dim)` final slot embeddings.
- `token_features`: `(B, 64, token_dim)` square-token sequence.
- `trunk_features`: `(B, channels, 8, 8)` CNN stem output.
- `relation_bias`: `(B, num_slots, 64)` relation bias the slots saw.
- `ablation_active`, `uses_relation_bias`, `uses_global_residual`,
  `uses_slot_attention`, `num_slots_levels`, `slot_iters_levels`:
  `(B,)` flags exposing the running ablation.

## Ablations

The packet's risk ("certificate slots may collapse") is exercised
through:

- `"none"` -- main model.
- `"no_relation_bias"` -- drop the chess relation prior, leaving the
  slots to discover structure from positional features only.
- `"no_global_residual"` -- read out from the slots only; tests
  whether the certificate is doing the work or piggy-backing on a
  global head.
- `"uniform_slot_attention"` -- replace softmax slot attention with
  uniform attention; tests whether the slots actually pick out a few
  squares.
- `"single_slot"` -- collapse to `num_slots = 1`; tests whether
  competition between certificate slots matters.

## Implementation Binding

- Registered model name: `forcing_certificate_transformer`
- Source implementation file: `src/chess_nn_playground/models/forcing_certificate_transformer.py`
- Idea-local wrapper: `ideas/i177_forcing_certificate_transformer/model.py`

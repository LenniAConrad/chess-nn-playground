# Architecture

`Set-Query Attention Bottleneck` is a board-only `puzzle_binary` classifier
that reads the board through a fixed-size bank of learned tactical queries.
It avoids a token-to-token transformer stack: square tokens do not freely mix
with each other. Instead, each query attends over the 64 board squares and the
classifier receives only attended values and attention-shape diagnostics.

## Tokenization

The model accepts the repository board tensor contract `B x 18 x 8 x 8`.
Each square becomes one token:

`token_i = MLP([board_planes_at_square_i, rank, file, centered_rank, centered_file, edge_distance, square_color])`.

The default config maps `model.channels` to `token_dim`, so the existing idea
config identity remains unchanged while still exposing the packet's token
dimension.

## Query Bottleneck

A learned query bank `Q x D` asks a small number of latent tactical questions.
Projected query, key, and value tensors are split into attention heads:

`a_{q,i,h} = softmax_i(q_{q,h}^T k_{i,h} / sqrt(d_h))`

`v_q = concat_h sum_i a_{q,i,h} value_{i,h}`

The exported `attention` tensor averages the heads and has shape `(B, Q, 64)`.
Rows sum to one in normal, frozen-query, and uniform/mean-pool ablation modes.

## Attention Diagnostics

For each query the model computes the packet diagnostics:

- attention entropy
- max attention
- best-second attention margin
- occupied and empty square mass
- side-to-move and opponent piece mass
- attended coordinate mean and variance

The prediction head classifies from the per-query attended values plus these
diagnostics. Central ablations are implemented by changing the classifier input
or attention source:

- `uniform_attention`: replace learned attention with uniform token averaging
- `random_frozen_queries`: keep the query bank fixed after initialization
- `value_only_no_diagnostics`: classify from attended values only
- `diagnostics_only`: classify from attention diagnostics only
- `mean_pool_matched_params`: use uniform set pooling in the same query-head
  surface

## Output Contract

The primary `logits` tensor has shape `(B,)` and is compatible with the
repository's BCE puzzle-binary trainer. The model also returns `attention`,
`attended_values`, `query_diagnostics`, and scalar summaries such as
`attention_entropy_mean`, `attention_margin_mean`, `occupied_attention_mass`,
`own_piece_attention_mass`, `opponent_piece_attention_mass`,
`attended_coord_rank_mean`, `attended_coord_file_mean`, `query_diversity`, and
`token_feature_energy`.

## Implementation Binding

- Registered model name: `set_query_attention_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/set_query_attention.py`
- Idea-local wrapper: `ideas/registry/i102_set_query_attention_bottleneck/model.py`

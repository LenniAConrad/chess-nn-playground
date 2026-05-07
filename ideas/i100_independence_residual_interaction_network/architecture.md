# Architecture

`Independence Residual Interaction Network` is a board-only `puzzle_binary`
classifier that removes a simple independence explanation before modeling
piece-square interaction evidence.

The model accepts the repository board tensor contract `B x 18 x 8 x 8`. It
uses the first 12 piece planes as the observed piece-square occupancy tensor
`x_piece` and uses the side-to-move plane only to build side-relative rank
marginals.

## Independence Baseline

For each board, the model computes safe current-board marginals:

- piece/channel counts and piece probabilities;
- occupied-square mass;
- side-relative rank marginals;
- file marginals.

It then builds two expected occupancy tensors:

`expected_direct(piece, square) = p(piece) * occupied(square)`

`expected_low_rank(piece, rank, file) = total_occupancy * p(piece) * p(rank) * p(file)`

The implemented expected tensor is a convex blend of these two product models
using `expected_mix` from config, defaulting to `0.5`:

`expected = expected_mix * expected_direct + (1 - expected_mix) * expected_low_rank`.

This keeps the baseline simple and explicit: it can explain material, occupied
squares, and coarse side-relative rank/file structure, but it cannot explain
piece-specific square interactions.

## Signed Residual Maps

The interaction signal is the signed residual:

`residual = x_piece - expected`.

The classifier receives residual maps, expected maps, the aggregate occupancy
map, and rank/file coordinate planes. A compact residual CNN encodes these maps,
and the final MLP combines pooled map features with deterministic summary
statistics:

- residual L1/L2 magnitude;
- positive and negative residual mass;
- maximum absolute residual;
- product-baseline mass ratio;
- piece, square, rank, file, and expected-map entropy;
- rank/file coupling left after the low-rank product;
- signed channel-coupling energy;
- material balance, center pressure, and occupancy count.

## Output Contract

The model returns a dictionary with `logits` as one BCE-compatible puzzle logit
tensor of shape `(B,)`. Diagnostics include `residual_l1`, `residual_l2`,
`positive_residual_mass`, `negative_residual_mass`, `max_abs_residual`,
`expected_mass_ratio`, `piece_entropy`, `square_entropy`, `rank_entropy`,
`file_entropy`, `expected_entropy`, `rank_file_coupling`,
`residual_signed_mean`, `interaction_energy`, `signed_channel_coupling`,
`material_balance`, `center_pressure`, and `occupancy_count`.

## Implementation Binding

- Registered model name: `independence_residual_interaction_network`
- Source implementation file: `src/chess_nn_playground/models/independence_residual.py`
- Idea-local wrapper: `ideas/i100_independence_residual_interaction_network/model.py`

# Architecture

`Replicator Payoff Piece Dynamics` models a position as a small differentiable
game over its occupied piece tokens. A learned pairwise payoff matrix split
across role heads drives a few replicator-dynamics steps; the resulting
per-head equilibrium / instability statistics are fused with a compact CNN
board summary and read out as a single ``puzzle_binary`` logit.

## Input And Token Construction

- Input is the repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- An `OccupiedPieceTokenizer` builds 23-feature raw tokens for every square:
  - 12-dim piece-type one-hot (own P,N,B,R,Q,K then opponent P,N,B,R,Q,K)
  - signed color flag (own=-1, opponent=+1, empty=0)
  - normalised file/rank coordinates centred on the board
  - Chebyshev distances to the own- and opponent-king centroids
  - side-to-move bit, castling availability scalar, en-passant flag
  - Manhattan- and Chebyshev-style centre proximity
  - occupancy flag
- A stable occupancy-descending sort keeps up to `max_pieces` (default `32`)
  squares per sample. The mask is `1` for occupied slots and `0` for padded
  slots; padding contributes no piece type or color and is suppressed in the
  replicator updates.

## Pairwise Payoff Matrix

- A 9-feature pair geometry table is precomputed once: `(file_j-file_i)/7`,
  `(rank_j-rank_i)/7`, normalised Chebyshev / Manhattan distances, same-file,
  same-rank, two same-diagonal flags, and a knight-step indicator. The
  tokenizer indexes this table by the selected square pairs to produce
  per-pair geometry for every batch element.
- A small MLP `f_theta` reads `[token_i, token_j, geometry_ij]` and emits one
  payoff scalar per role head, giving the asymmetric tensor
  `payoff: (batch, num_heads, max_pieces, max_pieces)`. Asymmetry is preserved
  so attack-like and defense-like roles can diverge.

## Replicator Dynamics

- Per-head initial logits are produced from the projected tokens by a linear
  `init_logits` layer, masked by occupancy and softmaxed to `(batch, num_heads,
  max_pieces)`.
- For `num_steps` iterations (default `5`) and per-head learnable step size
  `eta_h` (initialised to `0.5`), the model performs the standard log-domain
  replicator update

  ```text
  fitness_h_i = sum_j payoff_h_ij * p_h_j
  avg_h      = sum_i p_h_i * fitness_h_i
  log p_h    = log p_h + eta_h * (fitness_h - avg_h)
  log p_h    = log_softmax( log p_h + (mask - 1) * 1e9 )
  ```

  Padded slots remain pinned to ~0 mass at every step.

## Diagnostics And Head Fusion

- Per-head equilibrium statistics computed at the final population:
  - `entropy`, `top_mass`, KL divergence from the initial population,
    the average payoff `avg_payoff`, and a population-weighted fitness
    variance `fitness_variance`.
  - mass on own / opponent pieces, kings, pawns, minors, and majors derived
    by indexing the piece-type one-hot for the selected tokens.
- Global diagnostics: payoff-matrix asymmetry norm, total piece count, and a
  softmax over per-head pool weights so the classifier can mix heads.

## Board Encoder And Head

- A `BoardConvStem` (depth `2` by default) processes the simple_18 tensor and
  is mean+max pooled to a `2 * channels` board summary.
- The classifier concatenates the pooled summary with the flattened per-head
  statistics, the head-pool weights, the normalised piece count, and the
  normalised payoff-asymmetry norm. A two-layer MLP with GELU and dropout
  emits one logit for the `puzzle_binary` task (fine labels `0` and `1` map
  to non-puzzle, fine label `2` maps to puzzle).
- The forward pass returns a dict whose `logits` tensor has shape `(batch,)`
  alongside per-head and aggregated diagnostic tensors of shape `(batch,)`
  for ablation analysis.

## Implementation Binding

- Registered model name: `replicator_payoff_piece_dynamics`.
- Source implementation: `src/chess_nn_playground/models/replicator_payoff_piece_dynamics.py`.
- Idea-local wrapper: `ideas/registry/i131_replicator_payoff_piece_dynamics/model.py`.

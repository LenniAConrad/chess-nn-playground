# Architecture

## Architecture Description

The Rule-Consistent Latent Dynamics Network has:

1. board encoder
2. move descriptor encoder
3. latent transition model
4. auxiliary legal/reconstruction heads
5. puzzle classification head

The final model can run with only the current board at inference, or optionally with deterministic sampled legal move summaries.

## Input Format

```text
board: batch x 18 x 8 x 8
sampled_moves: batch x max_moves x move_feature_dim
sampled_next_boards: batch x max_moves x 18 x 8 x 8, training only
invalid_moves: batch x max_invalid x move_feature_dim, training only
```

## Forward Pass

```text
z = board_encoder(board)
move_tokens = move_encoder(sampled_moves)
next_latents_pred = transition(z, move_tokens)
legal_logits = legal_head(z, move_tokens_and_invalids)
dynamics_summary = summarize(next_latents_pred, legal_logits)
puzzle_logit = puzzle_head([z, dynamics_summary])
```

Training also encodes true next boards:

```text
z_next = board_encoder(next_board)
L_next = mse(next_latents_pred, stopgrad(z_next))
```

## Tensor Shapes

Suggested first version:

```text
latent_dim: 128
max_moves: 32
max_invalid: 32
move_feature_dim: 32
transition_layers: 2
output: batch x 1
```

## Output Heads

Primary:

```text
puzzle_logit
```

Auxiliary training heads:

```text
legal_move_logits
next_latent_prediction
next_board_piece_logits
```

Diagnostics:

```text
legal_entropy
transition_variance
max_transition_norm
```

## Parameter Estimate

```text
1M to 4M parameters
```

## FLOP Estimate

Main cost:

```text
O(batch * max_moves * latent_dim * transition_layers)
```

No full move tree is required.

## Implementation Binding

- Registered model name: `rule_consistent_latent_dynamics`.
- Source implementation: `src/chess_nn_playground/models/trunk/rule_dynamics.py`.
- Idea-local wrapper: `ideas/registry/i010_rule_consistent_latent_dynamics/model.py`.

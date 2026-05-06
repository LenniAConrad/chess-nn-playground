# Architecture

## Architecture Description

The model has three parts:

1. board encoder
2. deterministic action/reply token encoder
3. soft minimax pooling head

## Input Format

```text
board: batch x 18 x 8 x 8
actions: capped pseudo-legal side-to-move actions
replies: capped pseudo-legal replies for each action
```

Move tokens should include:

- from square
- to square
- piece type
- capture flag
- promotion flag
- check-like rule flag if available without search
- relation to kings and high-value pieces

## Forward Pass

```text
H = board_encoder(board)
U = encode_actions(actions, H)
V = encode_replies(replies, U, H)
action_promise = head_action(U)
reply_safety = head_reply(V)
minimax_scores = action_promise - softmax_pool(reply_safety)
global_minimax = softmax_pool(minimax_scores)
logit = classifier([pool(H), global_minimax, topk(minimax_scores)])
```

## Tensor Shapes

Suggested first version:

```text
max_actions: 48
max_replies_per_action: 24
token_dim: 96
output: batch x 1
```

## Output Heads

First benchmark:

```text
puzzle_logit
```

Optional diagnostics:

- best action index
- reply entropy
- top action-response gap

## Parameter Estimate

```text
1M to 3M parameters
```

depending on board trunk size.

## FLOP Estimate

The expensive part is action-reply token encoding:

```text
O(batch * max_actions * max_replies * token_dim)
```

Use shallow MLP encoders first. Do not rerun a full CNN for every action.

## Implementation Binding

- Registered model name: `response_minimax_classifier`.
- Source implementation: `src/chess_nn_playground/models/research_architectures.py`.
- Idea-local wrapper: `ideas/i002_response_minimax_classifier/model.py`.

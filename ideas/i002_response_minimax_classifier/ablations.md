# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `board_only` | Remove action/reply bottleneck. |
| `actions_only_no_replies` | Test whether side-to-move actions alone help. |
| `replies_count_only` | Use only reply counts/mobility. |
| `random_reply_tokens` | Destroy reply semantics while preserving shape. |
| `mean_pool_no_minimax` | Replace max-min pooling with average pooling. |

## What Each Ablation Tests

- `board_only`: measures total value of response modeling.
- `actions_only_no_replies`: tests whether opponent replies matter.
- `replies_count_only`: separates mobility from learned reply quality.
- `random_reply_tokens`: catches nonsemantic capacity gains.
- `mean_pool_no_minimax`: tests whether the min-max inductive bias matters.

## Falsification Criteria

Reject if:

```text
random_reply_tokens matches real replies
or mean_pool_no_minimax matches soft minimax
or board_only matches full model within noise across two seeds
```


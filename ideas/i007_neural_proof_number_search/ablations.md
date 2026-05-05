# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `depth_1` | Equivalent to shallow action scoring. |
| `depth_2` | One move plus opponent reply. |
| `depth_3` | Move, reply, continuation. |
| `mean_tree_pool` | Replace proof-number aggregation with mean pooling. |
| `no_and_or_roles` | Treat all nodes the same. |
| `random_legal_beam` | Preserve legal moves but remove tactical ordering. |
| `no_tree_trunk_only` | Board encoder only. |
| `bounded_context_zero` | Remove root context residual. |

## What Each Ablation Tests

- `depth_1/2/3`: tests whether multi-ply structure matters.
- `mean_tree_pool`: tests proof-number math.
- `no_and_or_roles`: tests game-tree role asymmetry.
- `random_legal_beam`: tests tactical beam quality.
- `no_tree_trunk_only`: tests total value over board-only classifier.
- `bounded_context_zero`: tests whether the tree can carry the prediction.

## Falsification Criteria

Reject if:

```text
depth_1 matches depth_3
or mean_tree_pool matches proof-number aggregation
or no_and_or_roles matches full model
or random_legal_beam matches tactical beam
```

Also reject if runtime is so high that a full benchmark is not practical unless accuracy gains are decisive.


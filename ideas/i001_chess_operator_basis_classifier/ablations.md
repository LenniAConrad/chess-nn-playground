# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `no_ray_operators` | Remove rank/file/diagonal line operators. |
| `no_jump_operators` | Remove knight/king/pawn adjacency operators. |
| `static_gates` | Replace input-conditioned gates with learned constants. |
| `cnn_local_only` | Replace all relation operators with local 3x3 convolution. |
| `shuffle_operator_masks` | Preserve edge counts but destroy chess geometry. |

## What Each Ablation Tests

- `no_ray_operators`: tests whether sliding-line geometry matters.
- `no_jump_operators`: tests whether non-convolutional piece movement matters.
- `static_gates`: tests whether context-dependent relation choice matters.
- `cnn_local_only`: tests whether the architecture is more than a CNN variant.
- `shuffle_operator_masks`: tests whether improvements come from chess geometry rather than extra edges.

## Falsification Criteria

Revise or reject the idea if:

```text
full model does not beat size-matched CNN PR AUC by >= 0.02
and no operator-family ablation changes near-puzzle FP by >= 0.02 absolute
```

Reject the novelty claim if shuffled operator masks perform the same as real chess masks.


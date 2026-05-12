# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `concat_fusion` | Concatenate branches and classify without agreement penalty. |
| `no_disagreement_penalty` | Keep factor logits but remove variance penalty. |
| `single_factor_grid` | Grid branch only. |
| `single_factor_piece` | Piece branch only. |
| `single_factor_relation` | Relation branch only. |
| `strong_residual_head` | Tests whether bypassing agreement erases the effect. |

## What Each Ablation Tests

- `concat_fusion`: tests whether agreement is better than ordinary fusion.
- `no_disagreement_penalty`: tests the specific bottleneck.
- single-factor ablations: identify which view carries signal.
- `strong_residual_head`: checks if the architecture relies on the agreement path.

## Falsification Criteria

Reject if:

```text
concat_fusion matches full model within noise
or near-puzzle mistakes do not show higher factor disagreement
or disagreement penalty reduces puzzle recall below 0.75 without PR AUC gain
```


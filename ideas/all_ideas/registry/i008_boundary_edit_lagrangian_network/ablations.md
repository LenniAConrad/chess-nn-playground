# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `base_only` | Remove edit solver. |
| `e_plus_only` | Use only puzzle-making energy. |
| `e_minus_only` | Use only puzzle-breaking energy. |
| `random_edit_basis` | Replace chess-shaped edits with random latent deltas. |
| `no_solver_direct_head` | Predict edit energies directly without unrolled optimization. |
| `single_edit_family` | Test each edit family alone. |
| `null_move_only` | Reduce to side-to-move edit only. |

## What Each Ablation Tests

- `base_only`: total value of boundary modeling.
- `e_plus_only` and `e_minus_only`: whether dual energies matter.
- `random_edit_basis`: chess edit semantics.
- `no_solver_direct_head`: whether optimization matters.
- `single_edit_family`: which edit family carries signal.
- `null_move_only`: whether this is just the null-move idea.

## Falsification Criteria

Reject if:

```text
base_only matches full model
or random_edit_basis matches legal edit basis
or no_solver_direct_head matches unrolled solver
or null_move_only matches the full edit basis
```

Also reject if edit energies do not separate source classes in the expected order:

```text
E_plus(random) > E_plus(near) > E_plus(puzzle)
```


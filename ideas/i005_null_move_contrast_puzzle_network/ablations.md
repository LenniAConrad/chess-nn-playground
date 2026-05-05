# Ablations

## Ablation Switches

| Switch | Meaning |
|---|---|
| `current_only` | Remove null view. |
| `concat_no_contrast` | Concatenate current/null latents but remove explicit delta. |
| `random_side_swap` | Use randomized side-to-move perturbations. |
| `stop_gradient_null` | Test whether null branch representation learning matters. |
| `no_positive_margin` | Remove optional positive-only margin. |

## What Each Ablation Tests

- `current_only`: measures total value of null contrast.
- `concat_no_contrast`: tests explicit contrast versus ordinary fusion.
- `random_side_swap`: tests deterministic chess counterfactuals.
- `stop_gradient_null`: tests whether shared contrast learning matters.
- `no_positive_margin`: tests whether the architecture alone is enough.

## Falsification Criteria

Reject if:

```text
current_only matches full model
or random_side_swap matches deterministic null view
or explicit delta provides no gain over concat_no_contrast
```

Also reject if the model improves accuracy only by reducing puzzle recall below the BT4 baseline region.


# Ablations

- Set `training.loss: bce_with_logits` with the same model.
- Set `contamination_dro.lambda_tail: 0.0`.
- Sweep `margin` and `beta`.
- Compare near-puzzle false-positive rate at matched puzzle recall.

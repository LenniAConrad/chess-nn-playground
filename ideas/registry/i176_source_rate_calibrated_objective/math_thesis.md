# Math Thesis

Source-Rate Calibrated Objective.

Source packet:
`ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`
(batch candidate rank `7`).

## Working thesis

The current benchmark's central question is not only "is F1 high?" It is:

```text
At useful puzzle recall, how many near-puzzles are falsely called puzzles?
```

Add a differentiable objective that penalises near-puzzle false-positive
rate at a target puzzle recall.

## Soft rates and rate-calibrated penalty

Let `p = sigmoid(logit)`. With a decision threshold `tau` and temperature
`temp`, the differentiable soft-rate indicator is:

```text
soft_indicator = sigmoid((logit - tau) / temp)
```

The conditional soft rates are:

```text
near_fp_soft        = mean over fine == 1 of soft_indicator
puzzle_recall_soft  = mean over fine == 2 of soft_indicator
```

The rate penalty is:

```text
loss_rate = lambda_fp     * relu(near_fp_soft - target_near_fp)^2
          + lambda_recall * relu(target_recall - puzzle_recall_soft)^2
```

and the final training loss is `BCEWithLogits + loss_rate`.

## Architectural realisation

To carry the thesis into the model itself rather than only the loss:

- `tau` and `temp` are learnable model parameters (with `temp` parameterised
  through `softplus` for positivity), so the calibration coevolves with the
  trunk.
- The puzzle logit is built from three explicit evidence channels - puzzle,
  near-puzzle, and random-negative - with non-negative weights on the
  negative channels. This makes "subtract near-puzzle evidence" a hard
  structural prior, matching the calibrated-objective failure mode being
  optimised.
- The forward output dict exposes `source_rate_soft_indicator`, `tau`, and
  `temp` directly so the rate penalty can be wired into any trainer without
  re-implementing the soft rates downstream.

The bespoke implementation of this thesis lives in
`src/chess_nn_playground/models/trunk/source_rate_calibrated_objective.py`; the
idea-local wrapper at `ideas/registry/i176_source_rate_calibrated_objective/model.py`
delegates to that source.

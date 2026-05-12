# Architecture

`Source-Rate Calibrated Objective` is a bespoke implementation of idea
`i176`. The packet thesis frames the puzzle_binary task as a *source-rate*
problem: at a target puzzle recall on fine label 2, how many fine-label-1
near-puzzles are falsely called puzzles? The proposal adds a differentiable
penalty on near-puzzle false-positive rate at a target puzzle recall. To
realise that thesis as architecture rather than a loss-only hack, this
folder ships a model that emits the calibration parameters as model state
and constructs the puzzle logit from explicitly named evidence channels.

## Pipeline

- Input: board tensor `(B, 18, 8, 8)`. CRTK / source metadata is
  reporting-only and never enters the forward pass.
- Convolutional trunk: `depth` blocks of `Conv2d(3x3) -> BatchNorm -> GELU
  (-> Dropout2d)`, lifting each square to `channels` features.
- Pooling: concat of channel-wise mean and max over the 8x8 grid produces a
  `2 * channels` feature vector.
- Three evidence heads share the pooled feature vector and emit one scalar
  each via `LayerNorm -> Linear -> GELU -> Dropout -> Linear`:
  - `puzzle_evidence` (fine label 2 / class "puzzle"),
  - `near_puzzle_evidence` (fine label 1 / hard non-puzzle), and
  - `random_negative_evidence` (fine label 0 / random non-puzzle).
- Negative-class weights are reparameterised through `softplus` so the
  calibrated logit always *subtracts* the negative-evidence channels from
  the puzzle channel:

  ```text
  logits = puzzle_evidence
         - softplus(near_weight_raw) * near_puzzle_evidence
         - softplus(random_weight_raw) * random_negative_evidence
  ```

- Calibration parameters: a learnable decision threshold `tau` and a
  learnable, strictly positive temperature `temp = softplus(temperature_raw)`
  drive the soft-rate sigmoid the calibrated objective consumes:

  ```text
  soft_indicator = sigmoid((logits - tau) / temp)
  near_fp_soft   = mean over fine == 1 of soft_indicator
  recall_soft    = mean over fine == 2 of soft_indicator
  ```

  The trainer reads `soft_indicator`, `tau` and `temp` straight off the
  forward output dict; nothing else is needed to wire the source-rate
  calibrated penalty alongside `BCEWithLogits`.
- Forward output dict keys: `logits`,
  `source_rate_puzzle_evidence`, `source_rate_near_puzzle_evidence`,
  `source_rate_random_negative_evidence`, `source_rate_threshold_tau`,
  `source_rate_temperature`, `source_rate_near_evidence_weight`,
  `source_rate_random_evidence_weight`, `source_rate_soft_indicator`.

## Why It Is Materially Distinct

- Three explicit per-fine-class evidence channels rather than a single
  pooled-feature classifier.
- `tau` and `temp` are learnable model parameters, not loss-time scalars,
  so the calibration is jointly trained with the trunk.
- Negative-channel weights are non-negative by construction, mirroring the
  rate-calibrated objective which directly penalises near-puzzle FP rate.

## Implementation Binding

- Registered model name: `source_rate_calibrated_objective` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/trunk/source_rate_calibrated_objective.py`
  (`SourceRateCalibratedObjectiveNetwork` and
  `build_source_rate_calibrated_objective_from_config`).
- Idea-local wrapper:
  `ideas/registry/i176_source_rate_calibrated_objective/model.py` calls
  `build_source_rate_calibrated_objective_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this idea.

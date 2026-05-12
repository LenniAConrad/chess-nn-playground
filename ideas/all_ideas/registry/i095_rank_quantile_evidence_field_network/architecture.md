# Architecture

`Rank-Quantile Evidence Field Network` realises the source packet's rank/quantile pooling thesis as a bespoke PyTorch model for the repository's `puzzle_binary` task. The classifier reads the position only through differentiable order-statistics of learned scalar evidence fields, so it has access to the full board but never gets a plain mean/sum-pooled summary.

## Implementation Binding

- Registered model name: `rank_quantile_evidence_field_network`
- Source implementation file: `src/chess_nn_playground/models/rank_quantile.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i095_rank_quantile_evidence_field_network/model.py`

## Modules

`EvidenceFieldEncoder` is a compact convolutional trunk over the simple_18 input. It optionally appends two coordinate planes (linear `rank` and `file` ramps in `[-1, 1]`), then runs `Conv2d -> GroupNorm -> GELU` blocks (with optional dropout) at width `channels` and finishes with a `1x1` projection to `evidence_fields` scalar fields `f_e: B x 8 x 8 -> R`. The encoder also keeps a deterministic random `Conv2d` weight bank and a deterministic `randperm(64)` square permutation as buffers — these are used by the `random_field_encoder` and `square_shuffle` ablations to test whether *learned* evidence fields and the underlying spatial layout matter.

`RankQuantilePooler` performs the differentiable rank/quantile read-out. For each evidence field `f_e` it:

- Sorts the 64 square activations to obtain order statistics `f_e^{(1)} <= ... <= f_e^{(64)}`.
- Linearly interpolates the configured quantile levels (default `0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99`) over those order statistics.
- Forms tail-gap features `q99 - q95`, `q95 - q50`, `q50 - q05`, `q05 - q01` to expose the proposal's *extreme*-evidence contrast.
- Computes per-field mean and standard deviation as capacity-matched mean-pool baselines that live alongside the rank features.
- Computes top-`k` and bottom-`k` mean order statistics for `k in (1, 4, 8)` as a witness-set readout that interpolates between extremes and a soft mean.
- Normalises a softmax over the centred field activations into a `rank_entropy` term, so the head can read *how concentrated* the evidence is across squares.
- Adds a robust range `q_max - q_min` and two soft-tail-mass scalars `sigmoid(8 * (f - q_max))`, `sigmoid(8 * (q_min - f))` averaged over squares.

The pooler returns a flat `[B, evidence_fields * (Q + 4 + 2 + 2K + 4)]` feature vector together with a diagnostic dictionary. The pool features are the *only* board-derived input the head sees; the head never reads the raw evidence map directly.

`RankQuantileEvidenceFieldNetwork` glues the trunk together: encoder -> pooler -> head. The head input concatenates the rank-quantile readout with a small `material_safe_stats` block of per-channel sums/means/maxes/mins plus a few aggregates — these are *board statistics*, not the rank features, and ensure that material-only ablations and the rank pooler can be capacity-matched. A `LayerNorm -> Linear -> GELU -> Dropout -> Linear -> GELU -> Linear(1)` MLP returns a single logit per board.

## Modes

The `mode` argument selects the active variant:

- `quantile` (default): full rank-quantile / tail-gap / soft tail-mass readout. The reference implementation called for in the source packet.
- `mean_pool_only`: replaces the per-field quantile and top-`k`/bottom-`k` slots with the per-field mean (broadcast to the same shape). Tests whether average board evidence is enough.
- `topk_only`: replaces the dense quantile slots with a single broadcast top-`k` mean and exposes the top-`k`/bottom-`k` block directly. Tests whether a narrow witness set captures the same signal as the full quantile curve.
- `random_field_encoder`: bypasses the learned trunk and pools rank features over fixed random `Conv2d` evidence fields. Tests whether *learned* fields matter for the rank/quantile signal.
- `square_shuffle`: applies a deterministic per-square permutation before encoding. Tests whether the spatial layout (and therefore any genuinely structural evidence pattern) is what the rank readout is exploiting.

The `mode_code` is exposed in the diagnostics so ablation harnesses can attach the active mode to each prediction.

## Diagnostics

`forward(x, *, return_fields=False)` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit `puzzle_binary` head.
- `prob`: sigmoid of the puzzle logit.
- `rank_features`: shape `(B, evidence_fields * per_field)`, the flat pool readout fed to the head.
- `material_safe_stats`: shape `(B, material_dim)`, the per-channel material/density block.
- `quantiles`: shape `(B, evidence_fields, Q)`, per-field interpolated quantiles.
- `tail_gaps`: shape `(B, evidence_fields, 4)`, the four tail-gap contrasts described above.
- `field_mean`, `field_std`: shape `(B, evidence_fields)`, per-field mean/std order summaries.
- `topk_means`, `bottomk_means`: shape `(B, evidence_fields, K)`, per-field top-`k`/bottom-`k` means.
- `rank_entropy`: shape `(B, evidence_fields)`, normalised softmax entropy of the per-square evidence.
- `robust_range`: shape `(B, evidence_fields)`, per-field robust quantile range.
- `high_tail_mass`, `low_tail_mass`: shape `(B, evidence_fields)`, soft tail mass above `q_max` / below `q_min`.
- `extreme_gap_mean`: shape `(B,)`, mean of upper and lower tail gaps across fields.
- `upper_tail_gap`, `lower_tail_gap`: shape `(B,)`, mean of the upper/lower tail gap across fields.
- `median_evidence`: shape `(B,)`, mean of the median quantile across fields.
- `max_quantile_evidence`, `min_quantile_evidence`: shape `(B,)`, max/min of the extreme quantiles.
- `field_energy`: shape `(B,)`, mean-squared field activation, a power proxy.
- `rank_readout_mode`: integer code identifying the active mode.
- `mechanism_energy`: `mean(tail_gaps^2)`, the order-statistic energy that operationalises the packet's `linear_algebra` mechanism family on the rank readout.
- `proposal_profile_strength`: per-board max of the upper extreme quantile across fields, a single-scalar proxy for the extreme-evidence signal.
- `proposal_keyword_count`: integer scalar preserved for compatibility with the project's research-packet diagnostic schema.

When `return_fields=True` the dict additionally contains `evidence_fields` of shape `(B, evidence_fields, 8, 8)` and `readout_features` (the head input) for ablation harnesses.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Model shapes: evidence fields `[B, evidence_fields, 8, 8]`, rank features `[B, evidence_fields * per_field]`, per-field quantiles `[B, evidence_fields, Q]`.
- The puzzle decision flows only through `psi(x) = [rank_pool(f) ; material_safe_stats(x)]` — the head never sees raw evidence maps directly, so the rank-quantile bottleneck is enforced architecturally.

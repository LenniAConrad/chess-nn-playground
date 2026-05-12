# Architecture

`Early-Exit Cascade BoardNet` realizes the source packet's cascade thesis as
a bespoke PyTorch model for the repo's `puzzle_binary` task: a shared
convolutional trunk fanned out into several classifier exits placed at
increasing depths, fused by a learned forward-halting expectation so a
single BCE-with-logits loss trains every exit.

## Implementation Binding

- Registered model name: `early_exit_cascade_boardnet`
- Source implementation file: `src/chess_nn_playground/models/trunk/early_exit_cascade_boardnet.py`
- Idea-local wrapper: `ideas/registry/i150_early_exit_cascade_boardnet/model.py`

## Modules

`EarlyExitCascadeBoardNet` accepts the project's `(B, 18, 8, 8)` board
tensor only. CRTK / source / engine / verification metadata is
reporting-only and is not consumed.

1. **Stem.** A `3x3` `Conv2d(input_channels -> channels)` followed by
   `BatchNorm2d` and `ReLU` lifts the board planes into a working channel
   dimension while preserving the `8 x 8` spatial layout.
2. **Cascade trunk.** `num_exits` sequential stages, each composed of
   `depth` `_ResidualBlock` units (two `3x3` `Conv2d` layers with
   `BatchNorm2d`, `ReLU`, and `Dropout2d`). Spatial size stays at `8 x 8`
   throughout so every exit sees the full board.
3. **Exit heads.** After each stage, an `_ExitHead` runs
   `AdaptiveAvgPool2d -> Flatten -> Linear(channels, hidden_dim) -> ReLU
   -> Dropout` and forks into a logit head and a halting head, each a
   `Linear(hidden_dim, 1)`.
4. **Forward-halting cascade.** Halting probabilities `h_k = sigma(halt_k /
   tau)` are read from the first `K - 1` exits. The cascade weight on exit
   `k < K - 1` is `w_k = h_k * prod_{j<k}(1 - h_j)`; the final exit takes
   the residual mass `w_{K-1} = prod_{j<K-1}(1 - h_j)`. The cascaded
   probability is `p = sum_k w_k * sigma(logit_k)` and the model emits
   `logit(p)` as `logits`.
5. **Numerical guards.** Halting and continuation log-probabilities are
   clipped by a small `prob_floor` before the cumulative-sum of
   `log(1 - h_j)`; the resulting weights are renormalised to defend against
   floating-point drift.

## Cascade fusion math

For exits indexed `k = 0, ..., K - 1` with logit `z_k` and halt score
`h_k = sigma(s_k / tau)` (only the first `K - 1` halt scores are used):

- `w_0 = h_0`
- `w_k = h_k * prod_{j<k}(1 - h_j)` for `0 < k < K - 1`
- `w_{K-1} = prod_{j=0}^{K-2}(1 - h_j)`

This is the discrete forward-halting distribution over a length-`K` cascade
and `sum_k w_k = 1` exactly (in real arithmetic). The cascaded probability
is `p = sum_k w_k * sigma(z_k)` and the model returns
`logit(p) = log(p / (1 - p))`. A standard BCE-with-logits loss on this
quantity differentiates with respect to every exit logit and every halt
score, so all exits learn jointly without any trainer modifications.

The auxiliary helper `cascade_multi_exit_loss` adds a per-exit BCE term
`(1 / K) * sum_k BCE(z_k, y)` weighted by `exit_weight`. It is exposed for
ablations and trainer wiring; the default trainer runs without it.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)`. BCE-compatible cascade log-odds for the
  one-logit `puzzle_binary` head.
- `logit`, `prob`: aliases of the cascade log-odds and probability.
- `exit_logits`: dict `{exit_0, ..., exit_{K-1}}` with per-exit logits of
  shape `(B,)`.
- `exit_probs`: per-exit sigmoid probabilities of shape `(B,)`.
- `exit_halt_logits`: per-exit raw halting scores; only the first `K - 1`
  are consumed by the cascade weights.
- `exit_logits_stack`, `exit_halt_stack`: stacked `(B, K)` tensors for
  vectorised diagnostics and ablation losses.
- `exit_weights`: shape `(B, K)`, the forward-halting weights `w_k` per
  example.
- `expected_exit_index`: shape `(B,)`, `sum_k k * w_k`. Low values mean
  the model is halting early on easy positions; high values mean it is
  refining ambiguous near-puzzles all the way to the last exit.
- `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`: scalars preserved for compatibility with the
  project's research-packet diagnostic schema.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. Engine, verification, source,
  CRTK, principal-variation, mate-score, and best-move metadata is
  reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed
  above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine
  label `2` maps to binary target `1`.
- Spatial layout is preserved at `8 x 8` for every exit so deeper exits
  do not lose access to the original board geometry.

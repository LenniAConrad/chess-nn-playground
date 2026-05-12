# Architecture

`Soft Majorization Line Sorter` is a bespoke implementation of idea
`i139`. It learns ``K`` scalar salience fields over the 8x8 board,
gathers their values along every chess line, soft-sorts each line in
descending order, and classifies the position from majorization-style
descriptors of those sorted profiles. Tactical content lives in *how
front-loaded a line is once sorted*, not in a bag-of-line statistic or a
token-routing attention map.

## Pipeline

- Input: board tensor `(B, 18, 8, 8)`. CRTK / source / engine metadata is
  reporting-only and never used as model input.
- Salience trunk: a compact `depth`-block CNN over the configured input
  channels followed by a 1x1 projection to `K` salience-head fields and a
  pooled `2 * channels`-dimensional board context vector
  (`mean(spatial) || max(spatial)`).
- Line geometry: the 46 standard chess lines (8 ranks + 8 files + 15
  diagonals + 15 anti-diagonals). Each line is materialized as a
  `(num_lines, 8)` flat-square index buffer plus a Boolean
  `(num_lines, 8)` validity mask so diagonals shorter than 8 are
  bookkept correctly.
- Differentiable sort: per (sample, salience-head, line) we apply the
  SoftSort operator of Prillo & Eisenschlos
  (`P = softmax(-|sort(s) - s| / tau)`, with the hard sort acting as a
  no-grad reference). Padded slots are pushed to the back with `-inf`,
  zeroed after sorting, and tracked by a length-aware `sort_mask` so they
  cannot leak into the majorization sums.
- Per-line majorization descriptors (11 features per
  `(sample, salience_head, line)`):
  - `top1`, `top2`, `top3` — soft-sorted top values
  - `gap01 = top1 - top2`, `gap12 = top2 - top3` — adjacent dominance gaps
  - `line_mean`, `line_sum`, `line_max_minus_mean` — Lp shape moments
  - `top1_concentration = |top1| / sum|s|`,
    `top2_concentration = (|top1| + |top2|) / sum|s|` — the canonical
    majorization concentration ratios
  - `normalized_entropy` — entropy of `softmax(soft_sorted)` divided by
    `log(L)`; low entropy = highly concentrated line
- Bucket pool: per `(salience_head, line_type)` bucket
  (line types = rank, file, diagonal, anti-diagonal) we keep the
  `mean` and `max` of every per-line descriptor. This produces a
  fixed-size `K * 4 * 22 = bucket_dim` summary that is
  invariant to how many lines a bucket has.
- Head: a LayerNorm + GELU MLP over
  `[board_context, bucket_descriptors_flat]` returning one puzzle logit.
- Diagnostics returned alongside `logits`: `smls_salience_fields`,
  `smls_line_values`, `smls_sorted_scores`, `smls_line_descriptors`,
  `smls_bucket_descriptors`, `smls_per_line_top1`,
  `smls_per_line_concentration`, `smls_per_line_gap01`,
  `smls_per_line_normalized_entropy`,
  `smls_mean_concentration_per_line_type`, `smls_most_active_line_type`,
  and `smls_board_context`.

## Why It Is Distinct

- Not the ray-language automaton (`ray_language_automaton_network`):
  there is no token grammar or finite-automaton state; the readout is a
  pooled sorted profile, not a recognized string.
- Not a state-space scan or recurrent line model: every line is read in
  closed form via SoftSort, with no per-line recurrence.
- Not Schur-Ray (`schur_ray_line_algebra_network`): no per-line linear
  solve.
- Not attention: `softmax(-|sort(s) - s| / tau)` permutes scalar salience
  values, it does not route tokens by content-based queries/keys.
- Not a generic CNN baseline: the head consumes sorted-profile
  majorization descriptors, not pooled CNN features alone — the board
  context is included additively so the linear-rank role of the sorted
  descriptors is testable end-to-end.

## Implementation Binding

- Registered model name: `soft_majorization_line_sorter` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/trunk/soft_majorization_line_sorter.py`
  (`SoftMajorizationLineSorter` and
  `build_soft_majorization_line_sorter_from_config`).
- Idea-local wrapper:
  `ideas/registry/i139_soft_majorization_line_sorter/model.py` calls
  `build_soft_majorization_line_sorter_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this idea.

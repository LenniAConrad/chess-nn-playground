# Implementation Notes

- Module location: `src/chess_nn_playground/models/primitives/pair_resonance_hessian_network.py`.
  Idea-local `model.py` calls the registry builder
  `build_pair_resonance_hessian_network_from_config`.

- Registry key: `pair_resonance_hessian_network`.

- Saliency: deterministic piece-value prior
  `(P=1, N=3, B=3.2, R=5, Q=9, K=0)` applied to the simple_18 piece
  occupancy planes. King value is 0 so the saliency stage cannot pick a
  king as a removal candidate (king removal is not chess-legal). The
  resulting top-K selection is computed from legal board state only —
  there is no CRTK metadata, source label, Stockfish score, or tactic tag
  in the saliency pipeline.

- Variant assembly: variant 0 is the unperturbed board. Variants
  `1..top_k` zero out one of the top-K piece occupancies each. Variants
  `top_k+1..top_k+pair_count` zero out the two pieces in the
  corresponding `itertools.combinations` index. Slots that point at an
  empty square (when fewer than `top_k` pieces exist) are masked-off via
  the `valid` flag and the corresponding pair-Hessian entries are zeroed
  before aggregation.

- PhiScorer: compact GroupNorm + GELU conv stack with mean+max pool and a
  two-layer scalar head. GroupNorm (not BatchNorm) is used so the same
  encoder can score multiple variant copies of the same board without
  leaking statistics across copies in a single batched forward pass.

- Gate initialisation: the final linear layer of the gate MLP has its bias
  initialised to `gate_init = -2.0`, which makes the sigmoid gate start
  near `0.12` so the primitive begins as a *small* additive correction to
  the i193 base logit. This matches the TSDP / TDCD primitive head pattern
  and keeps the optimisation honest — the head has to earn its contribution.

- Diagnostics: every per-sample scalar exported by the model uses the
  `(B,)` shape contract so the trainer's `_scalar_output_columns` helper
  surfaces them in `predictions_<split>.parquet`. The `dhpe_top_indices`
  tensor is `(B, top_k)` long; it is *not* a scalar column but is exposed
  for diagnostic dumps via the trainer's batched output dict.

- Cost: at the default `top_k=4`, every forward pass runs the PhiScorer on
  11 variants per position. Memory usage is dominated by the variant
  batch — for `B=128` and `top_k=4` it is roughly `128 * 11 * 18 * 8 * 8 *
  4 bytes = 6.5 MB` per batch, which is small compared to the trunk
  activations.

- Trainer extension: none required. The model returns the standard model
  output dict with `logits` and per-sample scalar diagnostics. The shared
  trainer's `_primary_logits` helper finds `logits`, and
  `_scalar_output_columns` surfaces the diagnostics. No new dataset
  columns, no new losses.

- Future precomputation hook: if the saliency stage needs to become
  learned (matching the spec's `s_i = |phi(P) - phi(P\i)|` description),
  the obvious extension is to pre-pass the variants through phi once for
  per-piece saliency and pick top-K from those scores. This would push
  the variant count to roughly `n + 1 + 2k(k-1)` per the spec. The
  current deterministic-saliency path is a documented choice that keeps
  the scout wall-clock to ~2.5x i193 instead of ~10x.

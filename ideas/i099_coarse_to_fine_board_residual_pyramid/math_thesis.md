# Math Thesis

Coarse-to-Fine Board Residual Pyramid

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `3`.

Working thesis: A puzzle-like position may be present in details not explained by coarse board summaries. Build a residual pyramid over the board: classify from what remains after each scale's coarse reconstruction explains the finer scale.

## Multi-scale residual decomposition

Let `x ∈ R^{C×8×8}` be the simple_18 board tensor and let `f_8 = stem(x ⊕ coords) ∈ R^{D×8×8}` be a learned feature map at the finest scale, where `coords ∈ R^{2×8×8}` are normalized rank/file planes. We define two coarse views by spatial averaging,

- `f_4 = down_{8 → 4}(avg_pool_2(f_8)) ∈ R^{D×4×4}`,
- `f_2 = down_{4 → 2}(avg_pool_2(f_4)) ∈ R^{D×2×2}`.

The pyramid then *predicts* each finer scale from the coarser scale and isolates what is not explained:

- `pred_4 = upsample_{2 → 4}(decode_{2 → 4}(f_2))`,
- `r_4 = refine_4(f_4 - pred_4)`, the residual at scale 4 (detail not explained by `f_2`),
- `expl_4 = pred_4 + α · r_4`,
- `pred_8 = upsample_{4 → 8}(decode_{4 → 8}(expl_4))`,
- `r_8 = refine_8(f_8 - pred_8)`, the residual at scale 8 (detail not explained by `expl_4`),
- `expl_8 = pred_8 + α · r_8`.

Here `α = residual_scale ≥ 0` is a scalar mixer (default `1.0`). When `α = 0` the model collapses to a coarse-only classifier, recovering the natural ablation.

## Classification head

The classifier consumes two streams that are forced to be complementary:

1. *Coarse stream* — pooled `(mean, max)` features of `f_2` concatenated with a deterministic 18-D simple_18 board summary (per-piece counts, side-to-move, signed material balance, occupancy count, rank/file/center pressure). This stream cannot see any fine detail by construction.
2. *Residual stream* — pooled `(mean, max)` features of `r_4` and `r_8`, augmented with scale-wise `L^1`, `L^2`, and `L^∞` residual statistics, the coarse `L^2` norm, an unexplained-energy ratio `||r_8||_2 / (||expl_8||_2 + ||r_8||_2)`, a residual-gain ratio `||r_8||_2 / ||r_4||_2`, and a sparsity proxy `||r_8||_∞ / ||r_8||_1`. These quantities only carry information that survived coarse reconstruction.

Final puzzle logit `s = classifier([head_coarse, head_residual]) ∈ R`. Setting `α = 0` removes the residual contribution from `expl_8` but keeps the residual diagnostics, so the residual head is testable in isolation while the coarse-only ablation is recovered exactly.

## Inductive claim

A puzzle-like position is hypothesised to be reflected in fine-scale board structure that the coarse summaries cannot reconstruct: the unexplained-energy ratio at scale 8 should be larger, the residual `r_8` should concentrate (`||r_8||_∞ / ||r_8||_1` larger), and the residual stream should add measurable signal beyond the coarse stream. Conversely, "non-puzzle" positions are expected to be well-explained by coarse summaries plus deterministic material/occupancy features.

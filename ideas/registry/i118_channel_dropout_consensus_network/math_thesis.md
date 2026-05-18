# Math Thesis

Channel Dropout Consensus Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `6`.

Working thesis: The classifier should not depend too heavily on one piece channel or artifact. Train a single shared encoder on deterministic channel-dropped board views and classify from consensus and disagreement features over those views.

## Formal description

Let `x ∈ R^{18 × 8 × 8}` be a simple_18 board tensor and let `M = {M_v}_{v=1}^{V}` be the set of `V = 6` deterministic channel-keep masks with `M_v ∈ {0, 1}^{18}`, with `M_1` (the full view) being all-ones and the remaining masks zeroing semantically grouped piece planes (pawns, minors, majors, white, black). The model defines:

- Channel-dropped views `x^{(v)} = x \odot M_v` (broadcasting `M_v` along the spatial axes).
- A shared encoder `Phi: R^{18 × 8 × 8} → R^{D × 8 × 8}` with one weight set, applied to every view.
- Per-view pooled latents `z_v = mean_{ij} Phi(x^{(v)})_{:, i, j} ∈ R^D`.
- Consensus / disagreement summaries
  - `mu = (1/V) ∑_v z_v` (mean across views),
  - `sigma^2 = (1/V) ∑_v (z_v - mu)^2` (per-feature population variance),
  - `delta = max_{i ≠ j} |z_i - z_j|` (per-feature max pairwise absolute distance),
  - `z_full = z_1` (full-view anchor latent).
- A puzzle logit `y = h([mu, sigma^2, delta, z_full])` where `h` is a LayerNorm + GELU MLP head.

The mask set `{M_v}` and the index of the full view (`v = 1`) are non-learnable constants. Only `Phi` and `h` are trainable.

## Decision rule

The packet's promotion criterion is: keep the model if the consensus / disagreement head beats both the `full_view_only` ablation and the `train_dropout_only` ablation on the puzzle_binary benchmark contract. Drop the model if `random_channel_masks` matches `none`, which would imply that the semantic view choice carried no signal beyond random per-view perturbations.

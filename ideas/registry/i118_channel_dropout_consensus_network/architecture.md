# Architecture

`Channel Dropout Consensus Network` is a bespoke shared-encoder ensemble whose head reads cross-view consensus *and* disagreement features.

## Implementation Binding

- Registered model name: `channel_dropout_consensus_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/channel_dropout_consensus.py`
- Idea-local wrapper: `ideas/registry/i118_channel_dropout_consensus_network/model.py`

- Mechanism family: `robustness`.
- Input: simple_18 board tensor only; CRTK/source metadata is reporting-only.
- Views: six deterministic channel-dropped variants of the input board are constructed by zeroing semantically grouped piece planes while leaving side-to-move, castling, and en-passant planes intact:
  - `full` — original board (used as the anchor view).
  - `remove_pawns` — zero piece planes `{0, 6}`.
  - `remove_minors` — zero piece planes `{1, 2, 7, 8}`.
  - `remove_majors` — zero piece planes `{3, 4, 9, 10}`.
  - `remove_white` — zero piece planes `{0..5}`.
  - `remove_black` — zero piece planes `{6..11}`.
- Shared trunk: a single convolutional encoder `Phi` (depth `depth`, width `channels`, optional BatchNorm) is applied to every view by stacking all `V` views along the batch axis, so one weight set sees every view in every forward pass.
- Per-view pooling: each view latent is mean-pooled over the 8x8 board to obtain `z_v ∈ R^D` with `D = channels`.
- Consensus / disagreement summaries:
  - `mean_latent = mean_v z_v` — consensus signal.
  - `variance_latent = var_v z_v` (population variance, per feature) — disagreement signal.
  - `max_pairwise = max_{i, j} |z_i - z_j|` (per feature) — worst-case disagreement.
  - `full_view_latent = z_{full}` — anchor latent.
- Head: `[mean_latent, variance_latent, max_pairwise, full_view_latent]` (shape `(B, 4D)`) is normed (LayerNorm), projected through a GELU MLP, and reduced to one puzzle logit. The head also returns `consensus_energy`, `disagreement_energy`, `max_pairwise_energy`, `full_view_energy`, and bookkeeping diagnostics (`mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`, `channel_dropout_ablation`, `channel_dropout_view_count`).

The view-masking buffers and the full-view index are non-learnable; only the shared encoder and the MLP head carry parameters. This makes the model materially distinct from a plain CNN baseline (no multi-view shared trunk pass), from ordinary channel dropout (no per-feature disagreement features in the head), and from the shared `ResearchPacketProbe` scaffold (no semantic channel-drop views at all).

## Supported ablations

`ChannelDropoutConsensusNetwork.ABLATIONS` enumerates the testable variants:

- `none` — full implementation as described above.
- `full_view_only` — encode only the full board; broadcast its latent across the view axis so the head sees `[full, 0, 0, full]`. Tests whether consensus across views is needed at all.
- `mean_only` — keep the multi-view encoder and the mean/full features but zero out the variance and max-pairwise disagreement features. Tests whether averaging suffices.
- `random_channel_masks` — replace the semantic drop-channel groups with fixed random piece-channel subsets of matched sizes (deterministic via `random_mask_seed`). Tests whether the semantic view choice matters.
- `train_dropout_only` — collapse to ordinary `nn.Dropout2d` channel dropout on the full board (no semantic views, no disagreement features). Tests whether explicit consensus features beat plain channel-dropout regularization.

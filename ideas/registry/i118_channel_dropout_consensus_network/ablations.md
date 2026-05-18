# Ablations

All ablations are exposed as the `model.ablation` config value and live on `ChannelDropoutConsensusNetwork.ABLATIONS`. They are runnable with the standard idea trainer and reported via `report_template.md`.

- `model.ablation: none` — full implementation: six deterministic views, shared encoder, mean + variance + max-pairwise + full-view head input.
- `model.ablation: full_view_only` — encode only the full board; mean and full-view features are tied; variance and max-pairwise features are zeroed. Tests whether consensus across views is needed at all (drop the consensus design if this matches `none`).
- `model.ablation: mean_only` — keep the multi-view encoder and the mean / full features but zero out the variance and max-pairwise disagreement features. Tests whether averaging across views suffices (drop the explicit disagreement features if this matches `none`).
- `model.ablation: random_channel_masks` — replace the semantic drop-channel groups (pawns, minors, majors, white, black) with deterministic random piece-channel subsets of matched sizes (seeded via `model.random_mask_seed`). Tests whether the semantic view choice carries signal beyond random per-view perturbations (drop the model if this matches `none`).
- `model.ablation: train_dropout_only` — replace the multi-view trunk with ordinary `nn.Dropout2d` channel dropout on the full board (rate `model.view_dropout_p`); at inference the layer is a no-op, so the model collapses to a single-view CNN classifier. Tests whether explicit consensus features beat plain channel-dropout regularization (drop the model if this matches `none`).
- Compare against the bespoke ideas in this registry (e.g., `i116_symmetric_difference_twin_encoder` for the twin-encoder family, `i066_bispectral_phase_coupling_board_network` for a non-CNN board operator), `simple_cnn` / `residual_cnn` baselines, LC0 BT4, and NNUE on the same split and seeds.

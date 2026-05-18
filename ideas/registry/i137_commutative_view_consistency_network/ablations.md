# Ablations

All ablations are exposed as the `model.ablation` config value and live on `CommutativeViewConsistencyNetwork.ABLATIONS`. They are runnable with the standard idea trainer and reported via `report_template.md`.

- `model.ablation: none` — full implementation: five view encoders, eight learned low-rank maps, nine defect feature vectors (six direct + three two-step cycles), and a head that reads view summaries plus per-defect MSE / mean-abs / signed-mean / max-abs / cosine statistics.
- `model.ablation: views_only_no_defects` — zero out every defect feature so the head reads only the five projected view summaries. Tests whether defect features add information beyond multi-view features (drop the consistency design if this matches `none`).
- `model.ablation: single_square_view` — disable the piece/line/region/count encoders by zeroing their latents so the head sees only the square latent (matched parameters). Tests whether the multi-view system is needed at all (drop the model if this matches `none`).
- `model.ablation: random_view_maps` — freeze the cross-view maps at deterministic random scale-matched values (seeded via `model.random_map_seed`). Tests whether the learned maps add information beyond fixed-scale residual regularizers (drop the model if this matches `none`).
- `model.ablation: count_to_all_only` — restrict every defect path to start from `z_count` by zeroing the other view latents before the maps are applied. Tests whether the model is collapsing to a material shortcut; if strong, report count-stratified metrics.
- `model.ablation: shuffled_piece_view` — permute the per-square piece tokens across the batch before the piece DeepSets encoder runs. Tests whether the piece view contributes real piece-square geometry; should degrade if the piece view encodes anything beyond piece counts.
- Compare against the bespoke ideas in this registry (e.g., `i118_channel_dropout_consensus_network` for the consensus-disagreement family, `i066_bispectral_phase_coupling_board_network` for a non-CNN board operator), `simple_cnn` / `residual_cnn` baselines, LC0 BT4, and NNUE on the same split and seeds.

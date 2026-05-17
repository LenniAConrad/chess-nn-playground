# Implementation Notes

- The model builder lives in this folder as
  `ideas/registry/a006_bt4_pareto_antichain_frontier_mixer/model.py` and
  exposes `build_model_from_config(config)` as required by the shared idea
  guard.
- The wrapper simply forwards to
  `chess_nn_playground.models.architecture.bt4_primitive_mixer.build_bt4_primitive_mixer_from_config`
  with `mixer=pareto_antichain_frontier` injected as the per-block spatial
  mixer.
- The mixer body lives at
  `src/chess_nn_playground/models/architecture/bt4_mixers/pareto_antichain_frontier.py`
  and is registered under the slug `pareto_antichain_frontier` via the
  `register_mixer` decorator in
  `src/chess_nn_playground/models/architecture/bt4_mixers/_base.py`.
- The registered model name `bt4_pareto_antichain_frontier_mixer` is an alias
  auto-registered by `chess_nn_playground.models.registry` from the
  `bt4_primitive_mixer` base plus the mixer slug; the config sets
  `model.name: bt4_pareto_antichain_frontier_mixer` and
  `model.mixer: pareto_antichain_frontier` so the registry and idea guard
  agree.
- Keep `idea.yaml`'s `idea_id`, `slug`, `implementation_kind: bespoke_model`,
  `implementation_status: implemented`, and `device: nvidia` aligned with
  `config.yaml` -- the shared scaffold validator in
  `chess_nn_playground.ideas.implementation.validate_idea_scaffold` rejects
  any disagreement.
- Do not edit the shared trainer, dataloader contract, or report pipeline;
  the only idea-specific surface in this folder is the wrapper plus the
  mixer module under `bt4_mixers/`.
- Forward-shape contract (smoked by `tests/test_idea_registry.py`): input
  `(B, 18, 8, 8)` simple_18 board tensor, output dict with `logits` of shape
  `(B,)`.
- The mixer is a learned surrogate for the p001 PAFR primitive's
  partial-order reducer. The default `(K, D, C_u) = (16, 48, 6)` keeps the
  pairwise dominance product at `O(B * K^2 * C_u) = O(B * 1536)` per block,
  cheap relative to the BT4 trunk pass. The per-square scatter-back path
  reuses the compiling attention `alpha_k in R^{B x K x 64}` so the
  partial-order math is what drives spatial mixing, not an unrelated
  output projection.

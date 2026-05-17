# Implementation Notes

- The model builder lives in this folder as
  `ideas/registry/a005_bt4_rule_aware_tactical_head_mixer/model.py` and
  exposes `build_model_from_config(config)` as required by the shared idea
  guard.
- The wrapper simply forwards to
  `chess_nn_playground.models.architecture.bt4_primitive_mixer.build_bt4_primitive_mixer_from_config`
  with `mixer=rule_aware_tactical_head` injected as the per-block spatial
  mixer.
- The mixer body lives at
  `src/chess_nn_playground/models/architecture/bt4_mixers/rule_aware_tactical_head.py`
  and is registered under the slug `rule_aware_tactical_head` via the
  `register_mixer` decorator in
  `src/chess_nn_playground/models/architecture/bt4_mixers/_base.py`.
- The registered model name `bt4_rule_aware_tactical_head_mixer` is an alias
  auto-registered by `chess_nn_playground.models.registry` from the
  `bt4_primitive_mixer` base plus the mixer slug; the config sets
  `model.name: bt4_rule_aware_tactical_head_mixer` and
  `model.mixer: rule_aware_tactical_head` so the registry and idea guard
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
- The mixer is a learned surrogate for the i248 TSDP primitive's rule-exact
  forcing features; the 8 depthwise direction convs (4 rook + 4 bishop) are
  initialised so each kernel reads its single-step neighbour cell, giving
  the optimiser a reasonable starting point that matches the geometry of
  check / capture / promotion rays.

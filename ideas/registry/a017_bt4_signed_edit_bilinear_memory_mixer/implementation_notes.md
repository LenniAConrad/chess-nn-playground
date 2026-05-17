# Implementation Notes

- The model builder lives in this folder as
  `ideas/registry/a017_bt4_signed_edit_bilinear_memory_mixer/model.py` and
  exposes `build_model_from_config(config)` as required by the shared idea
  guard.
- The wrapper simply forwards to
  `chess_nn_playground.models.architecture.bt4_primitive_mixer.build_bt4_primitive_mixer_from_config`
  with `mixer=signed_edit_bilinear_memory` injected as the per-block spatial
  mixer.
- The mixer body lives at
  `src/chess_nn_playground/models/architecture/bt4_mixers/signed_edit_bilinear_memory.py`
  and is registered under the slug `signed_edit_bilinear_memory` via the
  `register_mixer` decorator in
  `src/chess_nn_playground/models/architecture/bt4_mixers/_base.py`.
- The registered model name `bt4_signed_edit_bilinear_memory_mixer` is an
  alias auto-registered by `chess_nn_playground.models.registry` from the
  `bt4_primitive_mixer` base plus the mixer slug; the config sets
  `model.name: bt4_signed_edit_bilinear_memory_mixer` and
  `model.mixer: signed_edit_bilinear_memory` so the registry and idea guard
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
  `(B, 18, 8, 8)` simple_18 board tensor, output dict with `logits` of
  shape `(B,)`.
- The mixer is a static-position adaptation of the SEBM source primitive
  (`p012_signed_edit_bilinear_memory`). The 64 squares stand in for the
  source primitive's active piece-square feature set; per-square
  projections `a_j = A x_j` and `b_j = B x_j` feed the pair-state identity
  `p = s (.) u - sum_j a_j (.) b_j` that the source primitive maintains
  exactly under O(|Delta|) signed edits. The bilinear-rank hyperparameter
  `model.bilinear_rank` (default 64) controls the rank of the projections;
  smaller values cost less compute and reduce the rank of the pair memory.
- The FiLM broadcast layer (a 3r -> 2r linear that emits per-batch
  `gamma`/`beta`) is the per-token readout adaptation introduced to satisfy
  the shape-preserving spatial-mixer contract -- the source primitive emits
  only a single `(B, 3r)` vector with no per-square output, so the FiLM
  layer is the only non-source-faithful surface in the mixer.

# Implementation Notes

- The model builder lives in this folder as
  `ideas/registry/a008_bt4_reply_channel_capacity_mixer/model.py` and
  exposes `build_model_from_config(config)` as required by the shared idea
  guard.
- The wrapper simply forwards to
  `chess_nn_playground.models.architecture.bt4_primitive_mixer.build_bt4_primitive_mixer_from_config`
  with `mixer=reply_channel_capacity` injected as the per-block spatial
  mixer.
- The mixer body lives at
  `src/chess_nn_playground/models/architecture/bt4_mixers/reply_channel_capacity.py`
  and is registered under the slug `reply_channel_capacity` via the
  `register_mixer` decorator in
  `src/chess_nn_playground/models/architecture/bt4_mixers/_base.py`.
- The registered model name `bt4_reply_channel_capacity_mixer` is an alias
  auto-registered by `chess_nn_playground.models.registry` from the
  `bt4_primitive_mixer` base plus the mixer slug; the config sets
  `model.name: bt4_reply_channel_capacity_mixer` and
  `model.mixer: reply_channel_capacity` so the registry and idea guard
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
- The mixer is a learned surrogate for the p003 RCC primitive's
  Blahut-Arimoto channel-capacity solver. The default
  `(K, R, D, iters, tau) = (16, 16, 48, 24, 1.0)` keeps the per-block cost
  at `O(B * (K + R) * 64 * D)` for the candidate / reply attention pools,
  `O(B * K * R * D)` for the bilinear reply-logit table, and
  `O(iters * B * K * R)` for the unrolled Blahut-Arimoto iterations -- all
  cheap relative to the BT4 trunk pass. The per-square scatter-back path
  reuses the candidate-compiling attention `alpha_k in R^{B x K x 64}` so
  the capacity math is what drives spatial mixing, not an unrelated output
  projection.
- Numerical stability: `safe_transition = transition.clamp_min(1e-8)` and
  `marginal = (...).clamp_min(1e-8)` guard the log terms in the Blahut-
  Arimoto update from underflow on degenerate transition rows;
  `max(tau, 1e-6)` guards the conditional softmax against a learned
  `tau -> 0`. The damped softmax update (full replacement, no Polyak
  averaging) is stable across the 24 unrolled steps because the per-row
  log-ratio is bounded by `log(R)` whenever the transition rows are not
  pathological.

# Implementation Notes

- The model builder lives in this folder as
  `ideas/registry/a007_bt4_regret_saddlepoint_mixer/model.py` and
  exposes `build_model_from_config(config)` as required by the shared idea
  guard.
- The wrapper simply forwards to
  `chess_nn_playground.models.architecture.bt4_primitive_mixer.build_bt4_primitive_mixer_from_config`
  with `mixer=regret_saddlepoint` injected as the per-block spatial
  mixer.
- The mixer body lives at
  `src/chess_nn_playground/models/architecture/bt4_mixers/regret_saddlepoint.py`
  and is registered under the slug `regret_saddlepoint` via the
  `register_mixer` decorator in
  `src/chess_nn_playground/models/architecture/bt4_mixers/_base.py`.
- The registered model name `bt4_regret_saddlepoint_mixer` is an alias
  auto-registered by `chess_nn_playground.models.registry` from the
  `bt4_primitive_mixer` base plus the mixer slug; the config sets
  `model.name: bt4_regret_saddlepoint_mixer` and
  `model.mixer: regret_saddlepoint` so the registry and idea guard
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
- The mixer is a learned surrogate for the p002 RSP primitive's
  entropy-regularized saddle solver. The default
  `(K, R, D, iters) = (16, 16, 48, 24)` keeps the per-block cost at
  `O(B * (K + R) * 64 * D)` for pooling, `O(B * K * R * D)` for the
  bilinear payoff, and `O(iters * B * K * R)` for the unrolled saddle
  iterations, all cheap relative to the BT4 trunk pass. The per-square
  scatter-back path reuses the candidate-compiling attention
  `alpha_k in R^{B x K x 64}` so the saddle math is what drives spatial
  mixing, not an unrelated output projection.
- Numerical stability: `inv_tau_p = 1 / max(tau_p, 1e-6)` and `inv_tau_q`
  guard against degenerate temperatures; the damped iteration with
  `damp = 0.35` keeps the softmax fixed-point stable across the 24 unrolled
  steps even when `A` is near-degenerate early in training.

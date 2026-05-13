# Trainer Notes

- Use `train.py` as-is — it calls `idea_train_cli(__file__)`, which checks
  idea/config/model identity through `validate_idea_for_training` before
  starting the shared `train_from_config` trainer.

- The shared `puzzle_binary` BCE-with-logits loss is sufficient. The model
  emits the puzzle logit at `out["logits"]`, which the trainer's
  `_primary_logits` helper picks up automatically. No custom loss or
  scheduler is required.

- Batch size: the CAIO side path runs the encoder twice per forward
  (original + colour-flipped for conjugacy error) and computes a
  `(B, amplitude_dim, 64, 64)` complex outer product per relation. The
  default config uses `batch_size: 192` (down from i193's 256) to keep
  peak GPU memory comparable. You can raise it back to 256 on a 12 GB
  GPU.

- Cost: roughly **1.6x i193 wall-clock per epoch**. For the scout protocol
  (12 epochs, single seed, single 3070) this stays inside the per-primitive
  overnight envelope.

- Mixed precision: enabled (`mixed_precision: true`). The complex tensor
  arithmetic in CAIO is intentionally kept in `float32` for autograd
  stability. The trunk remains AMP-eligible.

- `torch.compile`: **disabled for the first scout run** per the spec's
  cautionary note about complex backward + compile combinations. The
  eager forward is the validation path. Once a scout run produces a
  decision, revisit `torch.compile` separately.

- Reliability tier: `paper_grade` — matches i193 and the other primitives.

- Diagnostics surfaced via the standard trainer pathway:
  - `base_logit`, `primitive_delta`, `primitive_delta_raw`,
    `primitive_gate`, `primitive_gate_applied`, `primitive_gate_logit`,
    `primitive_gate_entropy`, `primitive_contribution`.
  - `caio_constructive_mean`, `caio_destructive_mean`, `caio_curl_mean`,
    `caio_conjugacy_error`, `caio_amplitude_norm`.
  - All `trunk_*` keys re-export the i193 trunk diagnostics.

- Ablation runs: add `model.ablation: <mode>` to the config to swap the
  ablation. The allowed strings are listed in `architecture.md`. Each
  ablation needs its own run directory so the slice reports remain
  comparable to the i193 baseline.

- Resume: standard `training.resume_from` / `training.resume_run_dir`
  paths work because the model state dict is plain PyTorch state.
